import os
import pandas as pd
import numpy as np
from scipy.stats import hypergeom, combine_pvalues

'''default AA classes'''
AA_CLASSES = {
    "hydrophobic": set("AVLIMFWP"),
    "polar": set("STNQYC"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "aromatic": set("FWYH")
}

def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.
    不依賴 statsmodels。

    Parameters
    ----------
    p_values : array-like
        一組 raw p-values。

    Returns
    -------
    q_values : np.ndarray
        FDR-adjusted p-values。
    """
    p_values = np.asarray(p_values, dtype=float)

    m = len(p_values)
    order = np.argsort(p_values)
    ranked_p = p_values[order]

    # BH adjustment
    adjusted = ranked_p * m / np.arange(1, m + 1)

    # 保證 q-value 隨 rank 單調
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # q-value 不可超過 1
    adjusted = np.minimum(adjusted, 1.0)

    # 還原原始順序
    q_values = np.empty(m)
    q_values[order] = adjusted

    return q_values


def export_cross_structure_hypergeom_analysis(
    structures,
    comparisons_config,
    output_dir,
    filename_prefix="hypergeom",
    alternative="enrichment",
    aa_classes=None
):
    """
    進行三層分析：

    1. 每個 protein structure 個別進行 hypergeometric test
    2. 對同一 Comparison × AA_Class 的多個 structure p-values
       使用 Fisher's method 進行 cross-structure integration
    3. 對所有 combined p-values 使用 Benjamini-Hochberg FDR correction

    Parameters
    ----------
    structures : dict
        蛋白質結構資料，例如：
        structures["7VSI"]["protein"]
        structures["7VSI"]["binding_site"]
        structures["7VSI"]["interacting"]

    comparisons_config : list of tuple
        每個 tuple:
        (background_key, target_key, comparison_name)

    filename_prefix : str
        輸出檔案名稱前綴。

    alternative : str
        "enrichment"：
            檢驗 target 中該 AA class 是否富集
            P(X >= k)

        "depletion"：
            檢驗 target 中該 AA class 是否耗減
            P(X <= k)

    Returns
    -------
    df_raw : DataFrame
        每個 structure 的原始 hypergeometric test。

    df_combined : DataFrame
        Fisher combined p-value、FDR q-value、
        -log10(q) 等結果。

    heatmap_df : DataFrame
        可直接拿去畫 heatmap：
        row = AA class
        column = comparison
        value = -log10(FDR q-value)
    """

    if aa_classes is None:
        aa_classes = AA_CLASSES

    if alternative not in ["enrichment", "depletion"]:
        raise ValueError(
            'alternative 必須為 "enrichment" 或 "depletion"'
        )

    # =========================================================
    # Step 1
    # 每個 structure 個別做 hypergeometric test
    # =========================================================

    raw_records = []

    for bg_key, tg_key, comp_name in comparisons_config:

        for structure_name, data in structures.items():

            bg_seq = data[bg_key]
            tg_seq = data[tg_key]

            N = len(bg_seq)
            n = len(tg_seq)

            for cls, aa_set in AA_CLASSES.items():

                K = sum(aa in aa_set for aa in bg_seq)
                k = sum(aa in aa_set for aa in tg_seq)

                # -----------------------------
                # Hypergeometric test
                # -----------------------------

                if alternative == "enrichment":

                    # P(X >= k)
                    p_val = hypergeom.sf(
                        k - 1,
                        N,
                        K,
                        n
                    )

                else:

                    # P(X <= k)
                    p_val = hypergeom.cdf(
                        k,
                        N,
                        K,
                        n
                    )

                # -----------------------------
                # Observed enrichment ratio
                # -----------------------------

                bg_frac = K / N if N > 0 else np.nan
                tg_frac = k / n if n > 0 else np.nan

                if bg_frac > 0:
                    obs_ratio = tg_frac / bg_frac
                else:
                    obs_ratio = np.nan

                raw_records.append({
                    "Comparison_Type": comp_name,
                    "Structure": structure_name,
                    "AA_Class": cls,

                    "N_(Bg_Total)": N,
                    "K_(Bg_Class)": K,
                    "n_(Tg_Total)": n,
                    "k_(Tg_Class)": k,

                    "Background_Fraction": bg_frac,
                    "Target_Fraction": tg_frac,
                    "Observed_Ratio": obs_ratio,

                    "p_value": p_val
                })

    df_raw = pd.DataFrame(raw_records)

    # =========================================================
    # Step 2
    # Cross-structure integration
    #
    # 對每一組：
    #
    # Comparison × AA_Class
    #
    # 把所有 structure 的 p-value 用 Fisher method 合併
    # =========================================================

    combined_records = []

    grouped = df_raw.groupby(
        ["Comparison_Type", "AA_Class"],
        sort=False
    )

    for (comparison, aa_class), group in grouped:

        p_values = group["p_value"].values

        # 避免數值上出現 p = 0 導致 log(0)
        # scipy 的極小值即可，不影響實際 interpretation
        p_values_safe = np.clip(
            p_values,
            np.finfo(float).tiny,
            1.0
        )

        fisher_stat, combined_p = combine_pvalues(
            p_values_safe,
            method="fisher"
        )

        ratios = group["Observed_Ratio"].dropna()

        # 這幾個不是拿來做 hypothesis test，
        # 而是方便判斷跨 structure 的 effect direction 是否一致。
        median_ratio = (
            ratios.median()
            if len(ratios) > 0
            else np.nan
        )

        min_ratio = (
            ratios.min()
            if len(ratios) > 0
            else np.nan
        )

        max_ratio = (
            ratios.max()
            if len(ratios) > 0
            else np.nan
        )

        if alternative == "enrichment":
            n_same_direction = int((ratios > 1).sum())
        else:
            n_same_direction = int((ratios < 1).sum())

        combined_records.append({
            "Comparison_Type": comparison,
            "AA_Class": aa_class,

            "N_Structures": len(group),

            "Fisher_Statistic": fisher_stat,
            "Combined_p_value": combined_p,

            "Median_Observed_Ratio": median_ratio,
            "Min_Observed_Ratio": min_ratio,
            "Max_Observed_Ratio": max_ratio,

            "N_Structures_Same_Direction": n_same_direction
        })

    df_combined = pd.DataFrame(combined_records)

    # =========================================================
    # Step 3
    # Benjamini-Hochberg FDR correction
    #
    # 目前：
    # 5 AA classes × 3 comparisons = 15 tests
    # =========================================================

    df_combined["FDR_q_value"] = benjamini_hochberg(
        df_combined["Combined_p_value"].values
    )

    # 避免 log10(0)
    q_safe = np.clip(
        df_combined["FDR_q_value"].values,
        np.finfo(float).tiny,
        1.0
    )

    df_combined["minus_log10_FDR"] = -np.log10(q_safe)

    # conventional q < 0.05
    df_combined["FDR_significant_0.05"] = (
        df_combined["FDR_q_value"] < 0.05
    )

    # =========================================================
    # Step 4
    # 整理成 heatmap-ready matrix
    # =========================================================

    heatmap_df = df_combined.pivot(
        index="AA_Class",
        columns="Comparison_Type",
        values="minus_log10_FDR"
    )

    # 固定 row 順序
    aa_order = [
        "hydrophobic",
        "polar",
        "positive",
        "negative",
        "aromatic"
    ]

    heatmap_df = heatmap_df.reindex(aa_order)

    # =========================================================
    # Export CSV
    # =========================================================

    os.makedirs(output_dir, exist_ok=True)

    raw_filename = os.path.join(
        output_dir,
        f"{filename_prefix}_per_structure.csv"
    )

    combined_filename = os.path.join(
        output_dir,
        f"{filename_prefix}_combined_FDR.csv"
    )

    heatmap_filename = os.path.join(
        output_dir,
        f"{filename_prefix}_heatmap.csv"
    )

    df_raw.to_csv(raw_filename, index=False)
    df_combined.to_csv(combined_filename, index=False)
    heatmap_df.to_csv(heatmap_filename)

    print("Analysis completed.")
    print(f"Per-structure results: {raw_filename}")
    print(f"Combined + FDR results: {combined_filename}")
    print(f"Heatmap matrix: {heatmap_filename}")

    return df_raw, df_combined, heatmap_df
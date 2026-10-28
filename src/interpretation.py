import pandas as pd
import shap
import numpy as np

from scipy.stats import hypergeom, combine_pvalues


def draw_shap(input_dataset_path, out_of_fold_shap_mean_path, max_display_num = 21):
    # ============================================================
    # 1. Load original dataset
    # ============================================================

    data = pd.read_csv(input_dataset_path, sep="\t", index_col=0)
    X = data.iloc[:, 1:]

    # ============================================================
    # 2. Load aggregated OOF SHAP values
    # ============================================================

    mean_shap_df = pd.read_csv(out_of_fold_shap_mean_path, index_col=0 )

    # ============================================================
    # 3. Check samples and features
    # ============================================================

    assert set(mean_shap_df.index) == set(X.index), \
        "Sample IDs between X and SHAP do not match."

    assert set(mean_shap_df.columns) == set(X.columns), \
        "Feature names between X and SHAP do not match."


    # ============================================================
    # 4. Match exact order
    # ============================================================

    mean_shap_df = mean_shap_df.loc[X.index, X.columns]

    # ============================================================
    # 5. Build SHAP Explanation
    # ============================================================

    shap_values_oof = shap.Explanation(
        values=mean_shap_df.values,
        data=X.values,
        feature_names=X.columns
    )

    # ============================================================
    # 6. Beeswarm
    # ============================================================

    shap.plots.beeswarm(shap_values_oof, max_display=max_display_num)

    top_indices = np.argsort(np.abs(shap_values_oof.values).mean(axis=0))[
        ::-1
    ][:int(max_display_num)-1]

    shap_values_top = shap_values_oof[:, top_indices]

    shap.plots.beeswarm(shap_values_top, max_display=max_display_num)


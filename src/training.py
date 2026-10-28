import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from .features import (
    select_features
)


def train_model_repeated_stratified_rf(
    file_path,
    output_dir,
    output_dir_model,
    feature_set="1048",
    n_splits=5,
    n_repeats=10,
    random_state=87,
    count_shap=True
):

    # ============================================================
    # 1. Create output directories
    # ============================================================

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_model, exist_ok=True)

    importance_raw_dir = os.path.join(output_dir, "importance_raw")
    os.makedirs(importance_raw_dir, exist_ok=True)

    shap_raw_dir = os.path.join(output_dir, "shap_raw")
    os.makedirs(shap_raw_dir, exist_ok=True)

    # ============================================================
    # 2. Load data
    # ============================================================

    print("Loading data...")
    data = pd.read_csv(file_path, sep="\t", index_col=0)

    # First column = label
    # Remaining columns = features
    y = data.iloc[:, 0]
    X_full = data.iloc[:, 1:]

    X = select_features(X_full, feature_set=feature_set)

    print(f"Number of samples  : {X.shape[0]}")
    print(f"Number of features : {X.shape[1]}")

    print("\nClass distribution:")
    print(y.value_counts())

    # ============================================================
    # 3. Check binary classification
    # ============================================================

    unique_classes = np.sort(y.unique())

    if len(unique_classes) != 2:
        raise ValueError(
            f"This function expects binary classification, but found classes: {unique_classes}"
        )
    if not set(unique_classes).issubset({0, 1}):
        raise ValueError(
            f"Labels must be 0 and 1. Found classes: {unique_classes}"
        )

    # ============================================================
    # 4. Initialize Repeated Stratified K-Fold
    # ============================================================

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )
    total_models = n_splits * n_repeats

    print(f"Cross-validation strategy: Repeated Stratified {n_splits}-Fold × {n_repeats} repeats")
    print(f"Total model evaluations: {total_models}")

    # ============================================================
    # 5. Initialize performance result containers
    # ============================================================

    model_performance = []

    feature_importances_list = []

    sensitivity_list = []
    specificity_list = []
    precision_list = []
    accuracy_list = []
    # balanced_accuracy_list = []
    mcc_list = []
    average_precision_list = []
    roc_auc_list = []

    # ============================================================
    # 6. Initialize sample-level prediction results
    # ============================================================

    sample_results = pd.DataFrame(
        0, index=X.index,
        columns=["true_label", "train_times", "test_times", "correct_times"]
    )
    sample_results["true_label"] = y.values

    # ============================================================
    # 7. Initialize SHAP aggregation containers
    # ============================================================

    # Sum of SHAP values across out-of-fold predictions
    shap_sum = pd.DataFrame(0.0, index=X.index, columns=X.columns)

    # Sum of squared SHAP values
    shap_squared_sum = pd.DataFrame(0.0, index=X.index, columns=X.columns)

    # Number of OOF SHAP observations for each sample
    shap_count = pd.Series(0, index=X.index, dtype=int)

    # ============================================================
    # 8. Repeated Stratified K-Fold training
    # ============================================================

    for fold_index, train_idx, test_idx in enumerate(cv.split(X, y)):

        repeat_index = fold_index // n_splits
        fold_number = fold_index % n_splits + 1

        print(
            f"\nTraining model "
            f"{fold_index + 1}/{total_models} "
            f"(Repeat {repeat_index + 1}/{n_repeats}, "
            f"Fold {fold_number}/{n_splits})..."
        )

        # ========================================================
        # 9. Get train/test data
        # ========================================================

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # ========================================================
        # 10. Update sample train/test counts
        # ========================================================

        sample_results.loc[X_train.index, "train_times"] += 1
        sample_results.loc[X_test.index, "test_times"] += 1

        # ========================================================
        # 11. Define Random Forest
        # ========================================================

        model = RandomForestClassifier(
            bootstrap=True,
            class_weight="balanced",
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features="log2",

            random_state=(random_state + fold_index),

            n_jobs=2
        )

        # ========================================================
        # 12. Train model
        # ========================================================

        model.fit(X_train, y_train)

        # ========================================================
        # 13. Prediction on X_test
        # ========================================================

        y_pred_proba = model.predict_proba(X_test)
        positive_class_index = list(model.classes_).index(1)
        y_pred_proba_positive = y_pred_proba[:,positive_class_index]

        # Apply a consistent probability threshold
        y_pred = (y_pred_proba_positive >= 0.5).astype(int)

        # ========================================================
        # 14. Calculate performance metrics
        # ========================================================

        # Threshold-dependent metrics
        sensitivity = metrics.recall_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        )
        specificity = metrics.recall_score(
            y_test, y_pred,
            pos_label=0,
            zero_division=0
        )
        precision = metrics.precision_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        )
        accuracy = metrics.accuracy_score(y_test, y_pred)
        # balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)

        # Threshold-independent metrics
        average_precision = metrics.average_precision_score(
            y_test, y_pred_proba_positive
        )
        roc_auc = metrics.roc_auc_score(
            y_test, y_pred_proba_positive
        )

        # ========================================================
        # 15. Store performance
        # ========================================================

        model_performance.append({
            "model_index": fold_index,
            "repeat": repeat_index + 1,
            "fold": fold_number,

            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "accuracy": accuracy,
            # "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "average_precision": average_precision,
            "roc_auc": roc_auc
        })

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        accuracy_list.append(accuracy)
        # balanced_accuracy_list.append(balanced_accuracy)
        mcc_list.append(mcc)

        average_precision_list.append(average_precision)
        roc_auc_list.append(roc_auc)

        # ========================================================
        # 16. Update sample correct prediction counts
        # ========================================================

        correct_prediction = (y_pred == y_test).astype(int)

        sample_results.loc[X_test.index, "correct_times"] += correct_prediction

        # ========================================================
        # 17. Store probability prediction
        # ========================================================

        probability_column = (
            f"model_{fold_index}"
            f"_positive_probability"
        )

        sample_results[probability_column] = -1.0

        sample_results.loc[X_test.index, probability_column] = y_pred_proba_positive

        # ========================================================
        # 18. Store predicted class
        # ========================================================

        prediction_column = f"model_{fold_index}_prediction"

        sample_results[prediction_column] = -1

        sample_results.loc[X_test.index, prediction_column] = y_pred

        # ========================================================
        # 19. Calculate OUT-OF-FOLD SHAP values
        # ========================================================

        if count_shap:

            print("Calculating OOF SHAP values...")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            if isinstance(shap_values, list):
                shap_values_class_1 = (shap_values[positive_class_index])

            else:
                shap_values = (np.asarray(shap_values))

                if shap_values.ndim == 3:
                    shap_values_class_1 = (
                        shap_values[
                            :,
                            :,
                            positive_class_index
                        ]
                    )

                elif shap_values.ndim == 2:
                    shap_values_class_1 = (shap_values)

                else:
                    raise ValueError("Unexpected SHAP output shape: {shap_values.shape}")

        # ========================================================
        # 20. Convert this fold's SHAP into DataFrame
        # ========================================================

            shap_df = pd.DataFrame(
                shap_values_class_1,
                index=X_test.index,
                columns=X.columns
            )

        # ========================================================
        # 21. Save raw OOF SHAP values for this fold
        # ========================================================

            shap_output_path = (
                os.path.join(
                    shap_raw_dir,
                    (
                        f"shap_values_model_"
                        f"{fold_index}_"
                        f"repeat_{repeat_index + 1}_"
                        f"fold_{fold_number}.csv"
                    )
                )
            )

            shap_df.to_csv(
                shap_output_path,
                index_label="sample_index"
            )

            print(f"OOF SHAP saved to: {shap_output_path}")

    # ========================================================
    # 22. Aggregate SHAP values
    # ========================================================

            shap_sum.loc[X_test.index] += shap_df
            shap_squared_sum.loc[X_test.index] += (shap_df ** 2)
            shap_count.loc[X_test.index] += 1

        ### if count_shap = False
        else:
            print("SHAP skipped")

    # ========================================================
    # 23. Feature importance
    # ========================================================

    feature_names = X.columns.tolist()

    feature_importances = model.feature_importances_

    feature_importance_df = (
        pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        })
    )

    feature_importance_df = (
        feature_importance_df.sort_values(by="Importance", ascending=False)
    )

    feature_importance_df.to_csv(
        os.path.join(importance_raw_dir, f"feature_importance_{fold_index}.csv"),
        index=False
    )

    feature_importances_list.append(feature_importances)

    # ========================================================
    # 24. Save model
    # ========================================================

    model_path = os.path.join(
            output_dir_model,f"random_forest_model_{fold_index}.pkl"
    )

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # ============================================================
    # 25. Convert performance results
    # ============================================================

    model_performance_df = (pd.DataFrame(model_performance))

    model_performance_df.to_csv(
        os.path.join(output_dir,"model_performance.csv"), 
        index=False
    )

    # ============================================================
    # 26. Calculate prediction aggregation
    # ============================================================

    probability_columns = [
        col
        for col in sample_results.columns
        if col.endswith(
            "_positive_probability"
        )
    ]

    prediction_columns = [
        col
        for col in sample_results.columns
        if col.endswith(
            "_prediction"
        )
    ]

    # Replace -1 with NaN before aggregation
    probability_data = (
        sample_results[
            probability_columns
        ].replace(
            -1,
            np.nan
        )
    )

    prediction_data = (
        sample_results[
            prediction_columns
        ].replace(
            -1,
            np.nan
        )
    )

    # Mean predicted probability across repeats
    sample_results[
        "mean_positive_probability"
    ] = (
        probability_data.mean(
            axis=1
        )
    )


    # SD of predicted probability across repeats
    sample_results[
        "std_positive_probability"
    ] = (
        probability_data.std(
            axis=1,
            ddof=0
        )
    )

    # Fraction of OOF models predicting positive
    sample_results["positive_prediction_fraction"] = (
        prediction_data.mean(axis=1)
    )

    # Accuracy for each sample across repeats
    sample_results["correct_fraction"] = (
        sample_results["correct_times"] / sample_results["test_times"]
    )

    # ============================================================
    # 27. Save sample-level prediction results
    # ============================================================

    sample_results.to_csv(
        os.path.join(output_dir, "sample_results.csv"),
        index_label="sample_index"
    )

    # ============================================================
    # 28. Calculate aggregated OOF SHAP mean
    # ============================================================

    if count_shap:
        if (shap_count == 0).any():
            missing_samples = (shap_count[shap_count == 0].index.tolist())
            raise RuntimeError(
                f"Some samples have no OOF SHAP values: {missing_samples}"
            )

        mean_shap_df = shap_sum.div(shap_count,axis=0)

    # ============================================================
    # 29. Calculate aggregated OOF SHAP SD
    # ============================================================

        mean_squared_shap = shap_squared_sum.div(shap_count, axis=0)
        shap_variance = (mean_squared_shap - mean_shap_df ** 2)

        # turn negative values to 0
        shap_variance = shap_variance.clip(lower=0)
        std_shap_df = np.sqrt(shap_variance)

    # ============================================================
    # 30. Save aggregated SHAP results
    # ============================================================

        mean_shap_df.to_csv(
            os.path.join(output_dir, "out_of_fold_SHAP_mean.csv"),
            index_label="sample_index"
        )

        std_shap_df.to_csv(
            os.path.join(output_dir, "out_of_fold_SHAP_std.csv"),
            index_label="sample_index"
        )

        # Save number of SHAP observations
        shap_count_df = shap_count.to_frame(name="shap_count")

        shap_count_df.to_csv(
            os.path.join(output_dir, "out_of_fold_SHAP_count.csv"),
            index_label="sample_index"
        )

    # ============================================================
    # 31. Mean absolute SHAP per feature
    # ============================================================

        mean_absolute_shap = (mean_shap_df.abs().mean(axis=0))

        mean_absolute_shap_df = (
            pd.DataFrame({
                "Feature": mean_absolute_shap.index,
                "Mean_Absolute_OOF_SHAP":  mean_absolute_shap.values
            })
        )

        mean_absolute_shap_df = (
            mean_absolute_shap_df.sort_values(
                by="Mean_Absolute_OOF_SHAP",
                ascending=False
            )
        )

        mean_absolute_shap_df.to_csv(
            os.path.join(output_dir, "mean_absolute_OOF_SHAP_by_feature.csv"),
            index=False
        )
    
    ### if count_shap = False
    # else:
    #     print("shap skipped")

    # ============================================================
    # 32. Calculate final RF feature importance
    # ============================================================

    feature_importances_array = (np.array(feature_importances_list))

    mean_importances = (feature_importances_array.mean(axis=0))
    std_importances = (feature_importances_array.std(axis=0))

    final_feature_importance_df = (
        pd.DataFrame({
            "Feature": X.columns,
            "Mean Importance": mean_importances,
            "Std Importance": std_importances
        })
    )

    final_feature_importance_df = (
        final_feature_importance_df.sort_values(
            by="Mean Importance", ascending=False
        )
    )

    final_feature_importance_df.to_csv(
        os.path.join(output_dir, "final_feature_importance.csv"),
        index=False
    )

    # ============================================================
    # 33. Performance summary
    # ============================================================

    performance_data = {
        "sensitivity": np.array(sensitivity_list),
        "specificity": np.array(specificity_list),
        "precision": np.array(precision_list),
        "accuracy": np.array(accuracy_list),
        # "balanced_accuracy": np.array(balanced_accuracy_list),
        "mcc": np.array(mcc_list),

        "average_precision": np.array(average_precision_list),
        "roc_auc": np.array(roc_auc_list)
    }

    performance_summary = {"Metric": [], "Mean": [], "Std": []}

    for (metric_name, values) in performance_data.items():
        performance_summary["Metric"].append( metric_name )
        performance_summary["Mean"].append(values.mean())
        performance_summary["Std"].append(values.std())

    performance_summary_df = pd.DataFrame(performance_summary)

    performance_summary_df.to_csv(
        os.path.join(output_dir, "model_performance_summary.csv"),
        index=False
    )

    # ============================================================
    # 34. Print final results
    # ============================================================

    print("\n" + "=" * 10)
    print("Repeated Stratified K-Fold CV with OOF SHAP completed!")
    print("=" * 10)

    print(f"Number of folds   : {n_splits}")
    print(f"Number of repeats : {n_repeats}")
    print(f"Total evaluations : {total_models}")

    print("\nPerformance summary:")
    print(performance_summary_df.to_string(index=False))

    print("\nSample test frequency:")
    print(sample_results["test_times"].value_counts().sort_index())

    if count_shap:
        print("\nOOF SHAP frequency:")
        print(shap_count.value_counts().sort_index())

        print("\nExpected OOF SHAP count per sample: {n_repeats}")

        print("\nTop 20 features by mean absolute OOF SHAP:")
        print(mean_absolute_shap_df.head(20).to_string(index=False))

    print("\nOutput files saved to:")
    print(output_dir)

    print("\nModels saved to:")
    print(output_dir_model)

    print("\nTraining and SHAP analysis completed!")

    # ============================================================
    # 35. Return results
    # ============================================================

    # return (
    #     model_performance_df,
    #     sample_results,
    #     final_feature_importance_df,
    #     performance_summary_df,
    #     mean_shap_df,
    #     std_shap_df
    # )


def train_model_repeated_stratified_xgboost(
    file_path,
    output_dir,
    output_dir_model,
    feature_set="1048",
    n_splits=5,
    n_repeats=10,
    random_state=87
):

    # ============================================================
    # 1. Create output directories
    # ============================================================

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_model, exist_ok=True)

    importance_raw_dir = os.path.join(output_dir, "importance_raw")
    os.makedirs(importance_raw_dir, exist_ok=True)

    # ============================================================
    # 2. Check input
    # ============================================================

    if not file_path:
        print("未提供任何輸入資料檔案。")
        return

    # ============================================================
    # 3. Load data
    # ============================================================

    print("Loading data...")

    base_data = pd.read_csv(file_path, sep="\t", index_col=0)

    y_full = base_data.iloc[:, 0]
    X_all = base_data.iloc[:, 1:]
    X_full = select_features(X_all, feature_set=feature_set)

    print(f"Number of samples  : {X_full.shape[0]}")
    print(f"Number of features : {X_full.shape[1]}")

    print("\nClass distribution:")
    print(y_full.value_counts())

    # ============================================================
    # 4. Check binary classification
    # ============================================================

    unique_classes = np.sort(y_full.unique())

    if len(unique_classes) != 2:
        raise ValueError(
            f"This function expects binary classification, but found classes: {unique_classes}"
        )
    if not set(unique_classes).issubset({0, 1}):
        raise ValueError(f"Labels must be 0 and 1. Found classes: {unique_classes}")

    # ============================================================
    # 5. Repeated Stratified K-Fold
    # ============================================================

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    total_models = n_splits * n_repeats
    print(f"Cross-validation strategy: Repeated Stratified {n_splits}-Fold × {n_repeats} repeats")
    print(f"Total model evaluations: {total_models}")

    # ============================================================
    # 6. Initialize result containers
    # ============================================================

    model_performance = []
    feature_importances_list = []

    sensitivity_list = []
    specificity_list = []
    precision_list = []
    accuracy_list = []
    # balanced_accuracy_list = []
    mcc_list = []

    average_precision_list = []
    roc_auc_list = []

    # ============================================================
    # 7. Sample-level results
    # ============================================================

    sample_results = pd.DataFrame(
        0, index=base_data.index,
        columns=["true_label", "train_times", "test_times", "correct_times"]
    )
    sample_results["true_label"] = y_full.values

    # ============================================================
    # 8. Repeated Stratified K-Fold training
    # ============================================================

    for fold_index, (train_idx, test_idx) in enumerate(
        cv.split(X_full, y_full)
    ):

        repeat_index = fold_index // n_splits
        fold_number = fold_index % n_splits + 1

        print(
            f"Training model {fold_index + 1}/{total_models} "
            f"(Repeat {repeat_index + 1}/{n_repeats}, "
            f"Fold {fold_number}/{n_splits})..."
        )

        # --------------------------------------------------------
        # Train/test data
        # --------------------------------------------------------

        X_train = X_full.iloc[train_idx]
        X_test = X_full.iloc[test_idx]

        y_train = y_full.iloc[train_idx]
        y_test = y_full.iloc[test_idx]

        # --------------------------------------------------------
        # Update sample counts
        # --------------------------------------------------------

        sample_results.loc[X_train.index, "train_times"] += 1
        sample_results.loc[X_test.index, "test_times"] += 1

        # ========================================================
        # Calculate scale_pos_weight from training data
        # ========================================================

        negative_count = np.sum(y_train == 0)
        positive_count = np.sum(y_train == 1)

        if positive_count == 0:
            raise ValueError("Training set contains no positive samples.")

        scale_pos_weight = negative_count / positive_count

        print(
            f"  Training class distribution: "
            f"negative={negative_count}, "
            f"positive={positive_count}, "
            f"scale_pos_weight={scale_pos_weight:.4f}"
        )

        # ========================================================
        # 9. XGBoost model
        # ========================================================

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=500,
            max_depth=9,
            learning_rate=0.1,
            subsample=0.8,
            scale_pos_weight=scale_pos_weight,

            random_state=random_state + fold_index,

            n_jobs=2
        )

        # ========================================================
        # 10. Train
        # ========================================================

        model.fit(X_train, y_train)

        # ========================================================
        # 11. Prediction
        # ========================================================

        y_pred_proba = model.predict_proba(X_test)
        positive_class_index = list(model.classes_).index(1)

        y_pred_proba_positive = y_pred_proba[:, positive_class_index]

        # Apply a consistent probability threshold
        y_pred = (y_pred_proba_positive >= 0.5).astype(int)

        # ========================================================
        # 12. Performance metrics
        # ========================================================

        # Threshold-dependent metrics
        sensitivity = metrics.recall_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        )
        specificity = metrics.recall_score(
            y_test, y_pred,
            pos_label=0,
            zero_division=0
        )
        precision = metrics.precision_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        )
        accuracy = metrics.accuracy_score(y_test, y_pred)
        # balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)

        # Threshold-independent metrics
        average_precision = metrics.average_precision_score(
            y_test, y_pred_proba_positive
        )
        roc_auc = metrics.roc_auc_score(
            y_test, y_pred_proba_positive
        )

        # ========================================================
        # 13. Store performance
        # ========================================================

        model_performance.append({
            "model_index": fold_index,
            "repeat": repeat_index + 1,
            "fold": fold_number,

            "negative_train": negative_count,
            "positive_train": positive_count,
            "scale_pos_weight": scale_pos_weight,

            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "accuracy": accuracy,
            # "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,

            "average_precision": average_precision,
            "roc_auc": roc_auc
        })

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        accuracy_list.append(accuracy)
        # balanced_accuracy_list.append(balanced_accuracy)
        mcc_list.append(mcc)
        average_precision_list.append(average_precision)
        roc_auc_list.append(roc_auc)

        # ========================================================
        # 14. Correct prediction count
        # ========================================================

        correct_prediction = (y_pred == y_test).astype(int)
        sample_results.loc[X_test.index, "correct_times"] += correct_prediction

        # ========================================================
        # 15. Save probability and prediction
        # ========================================================

        probability_column = (f"model_{fold_index}_positive_probability")
        sample_results[probability_column] = -1.0

        sample_results.loc[X_test.index, probability_column] = y_pred_proba_positive

        prediction_column = f"model_{fold_index}_prediction"

        sample_results[prediction_column] = -1
        sample_results.loc[X_test.index, prediction_column] = y_pred

        # ========================================================
        # 16. XGBoost feature importance
        # ========================================================

        feature_names = X_full.columns.tolist()
        feature_importances = (model.feature_importances_)

        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        })

        feature_importance_df = (
            feature_importance_df.sort_values(
                by="Importance",
                ascending=False
            )
        )

        feature_importance_df.to_csv(
            os.path.join(
                importance_raw_dir,
                f"feature_importance_{fold_index}.csv"
            ),
            index=False
        )

        feature_importances_list.append(feature_importances)

        # ========================================================
        # 17. SHAP
        # ========================================================

        # print(
        #     f"  Calculating SHAP for model "
        #     f"{fold_index + 1}/{total_models}..."
        # )

        # explainer = shap.TreeExplainer(model)

        # shap_values = explainer.shap_values(
        #     X_test
        # )

        # --------------------------------------------------------
        # SHAP summary plot
        # --------------------------------------------------------

        # plt.figure()

        # shap.summary_plot(
        #     shap_values,
        #     X_test,
        #     show=False
        # )

        # plt.tight_layout()

        # plt.savefig(
        #     os.path.join(
        #         output_dir,
        #         f"shap_summary_plot_model_{fold_index}.png"
        #     ),
        #     dpi=300,
        #     bbox_inches="tight"
        # )

        # plt.close()

        # --------------------------------------------------------
        # Save SHAP values
        # --------------------------------------------------------

        # shap_df = pd.DataFrame(
        #     shap_values,
        #     columns=X_test.columns,
        #     index=X_test.index
        # )

        # shap_df.to_csv(
        #     os.path.join(
        #         output_dir,
        #         f"shap_values_model_{fold_index}.csv"
        #     )
        # )

        # ========================================================
        # 18. Save model
        # ========================================================

        model_path = os.path.join(output_dir_model, f"xgboost_model_{fold_index}.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # ============================================================
    # 19. Model performance CSV
    # ============================================================

    model_performance_df = pd.DataFrame(model_performance)

    model_performance_df.to_csv(
        os.path.join(output_dir, "model_performance.csv"),
        index=False
    )

    # ============================================================
    # 20. Sample results
    # ============================================================

    sample_results.to_csv(
        os.path.join(output_dir, "sample_results.csv"),
        index_label="sample_index"
    )

    # ============================================================
    # 21. Feature importance summary
    # ============================================================

    feature_importances_array = np.array(feature_importances_list)
    mean_importances = (feature_importances_array.mean(axis=0))
    std_importances = (feature_importances_array.std(axis=0))

    final_feature_importance_df = pd.DataFrame({
        "Feature": X_full.columns,
        "Mean Importance": mean_importances,
        "Std Importance": std_importances
    })

    final_feature_importance_df = (
        final_feature_importance_df.sort_values(
            by="Mean Importance",
            ascending=False
        )
    )

    final_feature_importance_df.to_csv(
        os.path.join(output_dir, "final_feature_importance.csv"),
        index=False
    )

    # ============================================================
    # 22. Performance summary
    # ============================================================

    performance_data = {
        "sensitivity": np.array(sensitivity_list),
        "specificity": np.array(specificity_list),
        "precision": np.array(precision_list),
        "accuracy": np.array(accuracy_list),
        # "balanced_accuracy": np.array(balanced_accuracy_list),
        "mcc": np.array(mcc_list),

        "average_precision": np.array(average_precision_list),
        "roc_auc": np.array(roc_auc_list)
    }

    performance_summary = {
        "Metric": [],
        "Mean": [],
        "Std": []
    }

    for metric_name, values in performance_data.items():

        performance_summary["Metric"].append(metric_name)
        performance_summary["Mean"].append(values.mean())
        performance_summary["Std"].append(values.std())

    performance_summary_df = pd.DataFrame(performance_summary)

    performance_summary_df.to_csv(
        os.path.join(output_dir, "model_performance_summary.csv"),
        index=False
    )

    # ============================================================
    # 23. Print results
    # ============================================================

    print("\n" + "=" * 10)
    print("Repeated Stratified K-Fold XGBoost completed!")
    print("=" * 10)

    print(f"Number of folds     : {n_splits}")
    print(f"Number of repeats   : {n_repeats}")
    print(f"Total evaluations   : {total_models}")

    print("\nPerformance summary:")
    print(performance_summary_df.to_string(index=False))

    print("\nSample test frequency:")
    print(sample_results["test_times"].value_counts().sort_index())

    print("\nTraining and evaluation completed!")

    # ============================================================
    # 24. Return results
    # ============================================================

    # return (
    #     model_performance_df,
    #     sample_results,
    #     final_feature_importance_df,
    #     performance_summary_df
    # )


def train_model_repeated_stratified_svm(
    file_path,
    output_dir,
    output_dir_model,
    feature_set="1048",
    n_splits=5,
    n_repeats=10,
    random_state=87
):

    # ============================================================
    # 1. Create directories
    # ============================================================

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_model, exist_ok=True)

    # ============================================================
    # 2. Check input
    # ============================================================

    if not file_path:
        print("未提供任何輸入資料檔案。")
        return

    # ============================================================
    # 3. Load data
    # ============================================================

    print("Loading data...")

    base_data = pd.read_csv(file_path, sep="\t", index_col=0)

    y_full = base_data.iloc[:, 0]
    X_all = base_data.iloc[:, 1:]
    X_full = select_features(X_all, feature_set=feature_set)

    print(f"Number of samples  : {X_full.shape[0]}")
    print(f"Number of features : {X_full.shape[1]}")

    print("\nClass distribution:")
    print(y_full.value_counts())

    # ============================================================
    # 4. Check binary classification
    # ============================================================

    unique_classes = np.sort(y_full.unique())

    if len(unique_classes) != 2:
        raise ValueError(f"This function expects binary classification, but found classes: {unique_classes}")
    if not set(unique_classes).issubset({0, 1}):
        raise ValueError(f"Labels must be 0 and 1. Found classes: {unique_classes}")

    # ============================================================
    # 5. Repeated Stratified K-Fold
    # ============================================================

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )

    total_models = n_splits * n_repeats

    print(f"Cross-validation strategy: Repeated Stratified {n_splits}-Fold × {n_repeats} repeats")
    print(f"Total model evaluations: {total_models}")


    # ============================================================
    # 6. Initialize result containers
    # ============================================================

    model_performance = []

    sensitivity_list = []
    specificity_list = []
    precision_list = []
    accuracy_list = []
    # balanced_accuracy_list = []
    mcc_list = []

    average_precision_list = []
    roc_auc_list = []

    # ============================================================
    # 7. Sample-level results
    # ============================================================

    sample_results = pd.DataFrame(
        0, index=base_data.index,
        columns=["true_label", "train_times", "test_times", "correct_times"]
    )

    sample_results["true_label"] = y_full.values

    # ============================================================
    # 8. Repeated Stratified K-Fold training
    # ============================================================

    for fold_index, (train_idx, test_idx) in enumerate(
        cv.split(X_full, y_full)
    ):

        repeat_index = fold_index // n_splits
        fold_number = fold_index % n_splits + 1

        print(
            f"Training model {fold_index + 1}/{total_models} "
            f"(Repeat {repeat_index + 1}/{n_repeats}, "
            f"Fold {fold_number}/{n_splits})..."
        )

        # --------------------------------------------------------
        # Train/test data
        # --------------------------------------------------------

        X_train = X_full.iloc[train_idx]
        X_test = X_full.iloc[test_idx]

        y_train = y_full.iloc[train_idx]
        y_test = y_full.iloc[test_idx]

        # --------------------------------------------------------
        # Update train/test frequency
        # --------------------------------------------------------

        sample_results.loc[X_train.index, "train_times"] += 1
        sample_results.loc[X_test.index, "test_times"] += 1

        # ========================================================
        # 9. SVM
        # ========================================================

        model = SVC(
            probability=True,
            kernel="rbf",
            C=10,
            gamma="scale",

            random_state=random_state + fold_index
        )

        # ========================================================
        # 10. Train
        # ========================================================

        model.fit(X_train, y_train)

        # ========================================================
        # 11. Prediction
        # ========================================================

        y_pred_proba = model.predict_proba(X_test)
        positive_class_index = list(model.classes_).index(1)
        y_pred_proba_positive = y_pred_proba[:, positive_class_index]

        # Apply a consistent probability threshold
        y_pred = (y_pred_proba_positive >= 0.5).astype(int)

        # ========================================================
        # 12. Performance metrics
        # ========================================================

        # Threshold-dependent metrics
        sensitivity = metrics.recall_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        )
        specificity = metrics.recall_score(
            y_test, y_pred,
            pos_label=0,
            zero_division=0
        )
        precision = metrics.precision_score(
            y_test, y_pred,
            pos_label=1,
            zero_division=0
        )
        accuracy = metrics.accuracy_score(y_test, y_pred)
        # balanced_accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
        mcc = metrics.matthews_corrcoef(y_test, y_pred)

        # Threshold-independent metrics
        average_precision = metrics.average_precision_score(
            y_test, y_pred_proba_positive
        )
        roc_auc = metrics.roc_auc_score(
            y_test, y_pred_proba_positive
        )

        # ========================================================
        # 13. Store performance
        # ========================================================

        model_performance.append({
            "model_index": fold_index,
            "repeat": repeat_index + 1,
            "fold": fold_number,

            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy,
            # "balanced_accuracy": balanced_accuracy,
            "mcc": mcc,
            "average_precision": average_precision,
            "roc_auc": roc_auc
        })

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        precision_list.append(precision)
        accuracy_list.append(accuracy)
        # balanced_accuracy_list.append(balanced_accuracy)
        mcc_list.append(mcc)
        average_precision_list.append(average_precision)
        roc_auc_list.append(roc_auc)

        # ========================================================
        # 14. Correct prediction count
        # ========================================================

        correct_prediction = (y_pred == y_test).astype(int)
        sample_results.loc[X_test.index, "correct_times"] += correct_prediction

        # ========================================================
        # 15. Save probability and prediction
        # ========================================================

        probability_column = f"model_{fold_index}_positive_probability"
        sample_results[probability_column] = -1.0
        sample_results.loc[X_test.index, probability_column] = y_pred_proba_positive

        prediction_column = f"model_{fold_index}_prediction"
        sample_results[prediction_column] = -1
        sample_results.loc[X_test.index, prediction_column] = y_pred

        # ========================================================
        # 16. Save model
        # ========================================================

        model_path = os.path.join(output_dir_model, f"svm_model_{fold_index}.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # ============================================================
    # 17. Model performance CSV
    # ============================================================

    model_performance_df = pd.DataFrame(model_performance)

    model_performance_df.to_csv(
        os.path.join(output_dir, "model_performance.csv"),
        index=False
    )

    # ============================================================
    # 18. Sample results
    # ============================================================

    sample_results.to_csv(
        os.path.join(output_dir, "sample_results.csv"),
        index_label="sample_index"
    )

    # ============================================================
    # 19. Performance summary
    # ============================================================

    performance_data = {
        "sensitivity": np.array(sensitivity_list),
        "specificity": np.array(specificity_list),
        "precision": np.array(precision_list),
        "accuracy": np.array(accuracy_list),
        # "balanced_accuracy": np.array(balanced_accuracy_list),
        "mcc": np.array(mcc_list),

        "average_precision": np.array(average_precision_list),
        "roc_auc": np.array(roc_auc_list)
    }

    performance_summary = {
        "Metric": [],
        "Mean": [],
        "Std": []
    }

    for metric_name, values in performance_data.items():

        performance_summary["Metric"].append(metric_name)
        performance_summary["Mean"].append(values.mean())
        performance_summary["Std"].append(values.std())

    performance_summary_df = pd.DataFrame(performance_summary)

    performance_summary_df.to_csv(
        os.path.join(output_dir, "model_performance_summary.csv"),
        index=False
    )

    # ============================================================
    # 20. Print results
    # ============================================================

    print("\n" + "=" * 10)
    print("Repeated Stratified K-Fold SVM completed!")
    print("=" * 10)

    print(f"Number of folds     : {n_splits}")
    print(f"Number of repeats   : {n_repeats}")
    print(f"Total evaluations   : {total_models}")

    print("\nPerformance summary:")
    print(performance_summary_df.to_string(index=False))

    print("\nSample test frequency:")
    print(sample_results["test_times"].value_counts().sort_index())

    print("\nTraining and evaluation completed!")

    # ============================================================
    # 21. Return results
    # ============================================================

    # return (
    #     model_performance_df,
    #     sample_results,
    #     performance_summary_df
    # )
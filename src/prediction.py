import os
import pickle
import numpy as np
import pandas as pd
import shap
from sklearn import metrics

from .features import (
    select_features
)


def call_model_rf(
    file_path,
    model_dir,
    output_dir,
    feature_set="1048",
    count_shap=False
):
    """
    Apply previously trained Random Forest models to an input dataset.

    For each model:
        1. Load the saved Random Forest model.
        2. Predict positive-class probabilities.
        3. Count how many models classify each sample as positive.
        4. Optionally calculate SHAP values for the positive class.

    Parameters
    ----------
    file_path : str
        Path to the input TSV file.

        Expected format:

            Name        Label   Feature1   Feature2   Feature3
            Sample1     1       0.5        1.2        0.8
            Sample2     0       0.7        1.5        0.9
            Sample3     1       0.6        1.3        0.7

        The first column is used as the sample index.
        The second column is treated as the true label.
        All remaining columns are used as model features.

    model_dir : str
        Directory containing saved Random Forest models (.pkl).

    output_dir : str
        Directory for prediction and SHAP output files.

    count_shap : bool, default=False
        Whether to calculate SHAP values for each model.

    Returns
    -------
    results : pd.DataFrame
        Integrated prediction results containing:
            - true_label
            - positive_times
            - positive probability predicted by each model

    Notes
    -----
    This function assumes binary classification with class labels 0 and 1.
    Class 1 is treated as the positive class.
    """

    # ============================================================
    # 1. Create output directory
    # ============================================================

    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 2. Load data
    # ============================================================

    print("Loading data...")

    data = pd.read_csv(file_path, sep="\t", index_col=0)

    # ------------------------------------------------------------
    # Input structure:
    #
    # data.iloc[:, 0]  -> true label
    # data.iloc[:, 1:] -> features
    # ------------------------------------------------------------

    y = data.iloc[:, 0]

    X_all = data.iloc[:, 1:]

    X = select_features(X_all, feature_set=feature_set)

    print(f"Number of samples  : {X.shape[0]}")
    print(f"Number of features : {X.shape[1]}")
    print()

    # ============================================================
    # 3. Find model files
    # ============================================================

    model_files = sorted([
        file_name
        for file_name in os.listdir(model_dir)
        if file_name.endswith(".pkl")
    ])

    num_models = len(model_files)

    if num_models == 0:
        raise FileNotFoundError(
            f"No .pkl model files found in: {model_dir}"
        )

    print(f"Number of models   : {num_models}")
    print()

    # ============================================================
    # 4. Initialize prediction results
    # ============================================================

    results = pd.DataFrame(index=X.index)

    results["true_label"] = y.values
    results["positive_times"] = 0

    # ------------------------------------------------------------
    # Create one probability column for every model
    # ------------------------------------------------------------

    for model_index in range(num_models):

        probability_column = (
            f"model_{model_index}_positive_probability"
        )

        results[probability_column] = 0.0

    # ============================================================
    # 5. Apply each Random Forest model
    # ============================================================

    for model_index, model_file in enumerate(model_files):

        print(
            f"Applying model "
            f"{model_index + 1}/{num_models}: "
            f"{model_file}"
        )

        # --------------------------------------------------------
        # Load model
        # --------------------------------------------------------

        model_path = os.path.join(model_dir, model_file)

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # ========================================================
        # 6. Check feature compatibility
        # ========================================================

        if hasattr(model, "feature_names_in_"):

            model_features = list(model.feature_names_in_)
            input_features = list(X.columns)

            if model_features != input_features:

                raise ValueError(
                    f"Feature mismatch for model: {model_file}\n"
                    f"The input features do not match the features "
                    f"used during model training."
                )

        # ========================================================
        # 7. Prediction
        # ========================================================

        y_pred_proba = model.predict_proba(X)

        # --------------------------------------------------------
        # Find the probability corresponding to positive class (1)
        # --------------------------------------------------------

        if 1 not in model.classes_:
            raise ValueError(
                f"Positive class 1 was not found in "
                f"model.classes_: {model.classes_}"
            )

        positive_class_index = (
            list(model.classes_).index(1)
        )

        positive_proba = y_pred_proba[
            :,
            positive_class_index
        ]

        # --------------------------------------------------------
        # Store positive-class probabilities
        # --------------------------------------------------------

        probability_column = f"model_{model_index}_positive_probability"
        results[probability_column] = positive_proba

        # --------------------------------------------------------
        # Count positive predictions
        #
        # Probability > 0.5 is treated as positive.
        # --------------------------------------------------------

        positive_prediction = (
            positive_proba > 0.5
        ).astype(int)

        results[
            "positive_times"
        ] += positive_prediction

        # ========================================================
        # 8. SHAP analysis
        # ========================================================

        if count_shap:

            print(
                f"Calculating SHAP values for "
                f"model {model_index}..."
            )

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)

            # ----------------------------------------------------
            # Different SHAP versions may return different formats.
            #
            # Older versions:
            #
            #     list[
            #         class_0_SHAP,
            #         class_1_SHAP
            #     ]
            #
            # Newer versions:
            #
            #     ndarray(
            #         n_samples,
            #         n_features,
            #         n_classes
            #     )
            # ----------------------------------------------------

            if isinstance(shap_values, list):

                shap_values_positive = (
                    shap_values[positive_class_index]
                )

            elif (
                isinstance(shap_values, np.ndarray)
                and shap_values.ndim == 3
            ):

                shap_values_positive = (
                    shap_values[
                        :,
                        :,
                        positive_class_index
                    ]
                )

            elif (
                isinstance(shap_values, np.ndarray)
                and shap_values.ndim == 2
            ):

                shap_values_positive = shap_values

            else:

                raise ValueError(
                    f"Unexpected SHAP value format: "
                    f"{type(shap_values)}, "
                    f"shape="
                    f"{getattr(shap_values, 'shape', None)}"
                )

            # ----------------------------------------------------
            # Convert SHAP values to DataFrame
            # ----------------------------------------------------

            shap_df = pd.DataFrame(
                shap_values_positive,
                index=X.index,
                columns=X.columns
            )

            # ----------------------------------------------------
            # Save SHAP values
            # ----------------------------------------------------

            shap_output_path = os.path.join(
                output_dir, 
                f"shap_values_model_{model_index}.csv"
            )

            shap_df.to_csv(
                shap_output_path, 
                index_label="sample_index"
            )

            print(
                f"SHAP values saved to: {shap_output_path}"
            )

    # ============================================================
    # 9. Calculate integrated prediction statistics
    # ============================================================

    probability_columns = [
        f"model_{model_index}_positive_probability"
        for model_index in range(num_models)
    ]

    results["mean_positive_probability"] = results[
        probability_columns
    ].mean(axis=1)

    results["std_positive_probability"] = results[
        probability_columns
    ].std(axis=1)

    results["positive_ratio"] = (
        results["positive_times"] / num_models
    )

    # ============================================================
    # 10. Save integrated results
    # ============================================================

    results_output_path = os.path.join(output_dir, "integrated_results.csv")

    results.to_csv(results_output_path, index_label="sample_index")

    # ============================================================
    # 11. Print final results
    # ============================================================

    print("\n" + "=" * 10)
    print("Random Forest inference completed!")
    print("=" * 10)

    print(f"Number of samples : {X.shape[0]}")
    print(f"Number of models  : {num_models}")
    print(f"SHAP analysis     : {count_shap}")

    print(f"\nIntegrated results saved to:")

    print(results_output_path)

    if count_shap:
        print(f"\nSHAP values saved to:")
        print(output_dir)

    print("\nPrediction completed!")

    # ============================================================
    # 12. Return results
    # ============================================================

    return results
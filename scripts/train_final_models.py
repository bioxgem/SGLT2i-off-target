from src.training import (
    train_model_repeated_stratified_rf,
    train_model_repeated_stratified_xgb,
    train_model_repeated_stratified_svm
)


def main(
    feature_set = "1048", ### 1048, ecfp, moiety, numeric
    model_type = "rf" ### rf, xgb, svm
):

    valid_feature_sets = ["1048", "ecfp", "moiety", "numeric"]
    valid_model_types = ["rf", "xgb", "svm"]

    if feature_set not in valid_feature_sets:
        raise ValueError(f"Invalid feature_set: {feature_set}")

    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type: {model_type}")

    for stage in ["first", "second"]:

        file_path=f"./data/{stage}_1048.txt"

        print(
            f"\nTraining final {model_type} model: {stage}, "
            f"using feature set: {feature_set}"
        )

        if model_type == "rf":
            train_model_repeated_stratified_rf(
                file_path=file_path,
                output_dir=f"./results/rf_{stage}_{feature_set}",
                output_dir_model=f"./models/rf_{stage}_{feature_set}",
                feature_set=feature_set,
                n_splits=5,
                n_repeats=10,
                random_state=87,
                count_shap=True
            )
        elif model_type == "xgb":
            train_model_repeated_stratified_xgb(
                file_path=file_path,
                output_dir=f"./results/xgb_{stage}_{feature_set}",
                output_dir_model=f"./models/xgb_{stage}_{feature_set}",
                feature_set=feature_set,
                n_splits=5,
                n_repeats=10,
                random_state=87
            )
        elif model_type == "svm":
            train_model_repeated_stratified_svm(
                file_path=file_path,
                output_dir=f"./results/svm_{stage}_{feature_set}",
                output_dir_model=f"./models/svm_{stage}_{feature_set}",
                feature_set=feature_set,
                n_splits=5,
                n_repeats=10,
                random_state=87
            )

if __name__ == "__main__":
    main()
    
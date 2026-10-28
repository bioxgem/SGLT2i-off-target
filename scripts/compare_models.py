from src.training import (
    train_model_repeated_stratified_rf,
    train_model_repeated_stratified_xgboost,
    train_model_repeated_stratified_svm,
)


def main(
    stages = ["first", "second"],
    feature_set = "1048"
):

    stages = stages
    feature_set = feature_set

    for stage in stages:

        file_path = (f"./data/{stage}_{feature_set}.txt")

        print(f"\n===== {stage}: RF =====")
        train_model_repeated_stratified_rf(
            file_path=file_path,
            output_dir=f"./results/rf_{stage}_{feature_set}",
            output_dir_model=f"./models/rf_{stage}_{feature_set}",
            feature_set=feature_set,
            n_splits=5,
            n_repeats=10,
            random_state=87,
            count_shap=False
        )

        print(f"\n===== {stage}: XGBoost =====")
        train_model_repeated_stratified_xgboost(
            file_path=file_path,
            output_dir=f"./results/xgb_{stage}_{feature_set}",
            output_dir_model=f"./models/xgb_{stage}_{feature_set}",
            feature_set=feature_set,
            n_splits=5,
            n_repeats=10,
            random_state=87
        )

        print(f"\n===== {stage}: SVM =====")
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

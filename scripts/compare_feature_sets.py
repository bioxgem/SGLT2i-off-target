from src.training import ( 
    train_model_repeated_stratified_rf
)


def main(
    stages = [
        "first",
        "second"
    ],
    feature_sets = [
        "1048",
        "ecfp",
        "moiety",
        "numeric"
    ]
):

    stages = stages
    feature_sets = feature_sets

    for stage in stages:

        file_path = f"./data/{stage}_1048.txt"

        for feature_set in feature_sets:

            print("=" * 10)
            print(f"Stage: {stage} | Feature set: {feature_set}")
            print("=" * 10)

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

if __name__ == "__main__":
    main()

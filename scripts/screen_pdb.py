from src.prediction import (
    call_model_rf
)


def main(
    input_data = "pdb_valid_HAMW_filtered",
    model_type = "rf",
    feature_set = "1048"
):

    input_data = input_data
    model_type = model_type
    feature_set = feature_set

    for stage in ["first", "second"]:

        print(f"\nScreening PDB ligands: {stage}")

        call_model_rf(
            file_path=f"./data/{input_data}.txt",
            model_dir=f"./models/{model_type}_{stage}_{feature_set}",
            output_dir=f"./results/{model_type}_{stage}_{feature_set}_{input_data}",
            feature_set=feature_set,
            count_shap=False
        )

if __name__ == "__main__":
    main()

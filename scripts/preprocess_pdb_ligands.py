from src.preprocessing import (
    check_via_graphql
)


def main(
    input_path = (
        "./data_processing/"
        "HA_MW_filtered_ligands.txt"
    ),
    valid_output = (
        "./data_processing/"
        "valid_HA_MW_filtered_ligands.txt"
    ),
    invalid_output = (
        "./data_processing/"
        "invalid_HA_MW_filtered_ligands.txt"
    )
):

    input_path = input_path
    valid_output = valid_output
    invalid_output = invalid_output

    with open(input_path) as f:

        input_ids = [
            line.strip().upper()
            for line in f
            if line.strip()
        ]

    active_ligands = check_via_graphql(input_ids)

    input_set = set(input_ids)
    valid_set = set(active_ligands)

    invalid_ligands = sorted(input_set - valid_set)

    with open(valid_output, "w") as f:
        f.write("\n".join(active_ligands))

    with open(invalid_output, "w") as f:
        f.write("\n".join(invalid_ligands))

    print(f"num of input ligands: {len(input_ids)}")
    print(f"num of valid ligands: {len(active_ligands)}")
    print(f"num of invalid ligands: {len(invalid_ligands)}")

if __name__ == "__main__":
    main()

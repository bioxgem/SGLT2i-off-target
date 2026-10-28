from src.mapping import (
    analyze_specified_ligands
)


def main():

    TXT_FILE = "./data/pdb_ligand_first_passed.txt"
    JSON_FILE = "./data/pdb_ligand_uniprot_map.json"
    OUTPUT_CSV = "./data/pdb_ligand_first_passed_matched_ligand_uniprot_pairs.csv"

    analyze_specified_ligands(JSON_FILE, TXT_FILE, OUTPUT_CSV)

if __name__ == "__main__":
    main()

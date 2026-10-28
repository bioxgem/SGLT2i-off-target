import pandas as pd
import numpy as np

from src.statistics import (
    export_cross_structure_hypergeom_analysis
)


def main(
    input_file_path="./data/sglt2_structures.tsv",
    output_dir="./results/sglt2_aa_enrichment",
    filename_prefix="SGLT2_AA_enrichment",
    alternative="enrichment"
):

    df = pd.read_csv(input_file_path, sep="\t")

    structures = {}

    for structure_name, group in df.groupby("structure"):

        structures[structure_name] = dict(
            zip(group["region"], group["sequence"])
        )

    comparisons_config = [
        (
            "protein",
            "binding_site",
            "Binding_site_vs_Whole_protein"
        ),
        (
            "protein",
            "interacting",
            "Interacting_vs_Whole_protein"
        ),
        (
            "binding_site",
            "interacting",
            "Interacting_vs_Binding_site"
        )
    ]

    export_cross_structure_hypergeom_analysis(
        structures=structures,
        comparisons_config=comparisons_config,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        alternative=alternative,
        aa_classes=None
    )

if __name__ == "__main__":
    main()

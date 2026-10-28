# main workflow

# 1. Feature-set comparison
python -m scripts.compare_feature_sets

# 2. Model comparison
python -m scripts.compare_models

# 3. Train models for downstream screening
python -m scripts.train_final_models

# 4. PDB ligand screening
python -m scripts.screen_pdb


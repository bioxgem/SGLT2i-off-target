# SGLT2i-off-target
#### Workflow
![Workflow](/workflow.png "workflow")

# Directory Structure
├── README.md \
├── main.ipynb \
├── data/ \
│   ├── first_1048.txt \
│   ├── second_1048.txt \
│   └── pdb_valid_HAMW_filtered.txt \
├── scripts/ \
│   ├── compare_feature_sets.py \
│   ├── compare_models.py \
│   ├── train_final_models.py \
│   └── screen_pdb.py \
└── src/ \
    ├── features.py \
    ├── training.py \
    ├── prediction.py \
    ├── interpretation.py \
    └── mapping.py \

# Requirements
[pandas](https://pypi.org/project/pandas/) \
[numpy](https://pypi.org/project/numpy/) \
[scikit-learn](https://scikit-learn.org/stable/) \
[xgboost](https://pypi.org/project/xgboost/) \
[scipy](https://pypi.org/project/scipy/) \

import rdkit
from rdkit import Chem
from rdkit.Chem import inchi
import subprocess
import numpy as np
from rdkit.Chem import AllChem
import glob
from pandas.core.frame import DataFrame
import os
from pathlib import Path


# filter different feature sets from 1048

def select_features(X, feature_set="1048"):

    feature_set = feature_set.lower()

    expected_counts = {
        "1048": 1048,
        "ecfp": 512,
        "moiety": 519,
        "numeric": 17
    }

    if feature_set == "1048":

        X_selected = X.copy()

    elif feature_set == "ecfp":

        X_selected = X.loc[
            :,
            X.columns.str.startswith("ecfp_")
        ]

    elif feature_set == "moiety":

        X_selected = X.loc[
            :,
            X.columns.str.startswith(
                ("cm_", "pc_", "rd_")
            )
        ]

    elif feature_set == "numeric":

        X_selected = X.loc[
            :,
            X.columns.str.startswith(
                ("ac_", "stereo_")
            )
        ]

    else:

        raise ValueError(
            f"Unknown feature_set: {feature_set}. "
            f"Choose from: {list(expected_counts)}"
        )

    expected = expected_counts[feature_set]
    actual = X_selected.shape[1]

    if actual != expected:
        raise ValueError(
            f"Feature count mismatch for '{feature_set}': "
            f"expected {expected}, found {actual}."
        )

    print(
        f"Feature set       : {feature_set}\n"
        f"Selected features : {actual}"
    )

    return X_selected

def mol_to_ecfp(mol_file):
    mol = Chem.MolFromMolFile(mol_file, sanitize=True)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    return list(fp)

def mol_to_checkmol(mol_file):
    cmfp = np.zeros(204, dtype=int)
    checkmol_path = Path(__file__).resolve().parent / "checkmol"
    result = subprocess.run(
        [str(checkmol_path), "-p", str(mol_file)],
        capture_output=True, text=True, check=True
    )
    for line in result.stdout.strip().split('\n'):
        if ':' in line and '#' in line:
            try:
                parts = line.split(':')
                if int(parts[1].strip()) >= 1:
                    fea_num = int(parts[0].split('#')[1])
                    if 1 <= fea_num <= 204:
                        cmfp[fea_num - 1] = 1
            except (ValueError, IndexError):
                continue
    return cmfp

def mol_to_pubchem(mol_file):
    pcfp = np.zeros(168, dtype=int)
    src_dir = Path(__file__).resolve().parent
    matchmol_path = src_dir / "matchmol"
    pubchemFP_dir = src_dir / "PCFP_168"
    pubchemFP = sorted(pubchemFP_dir.glob("*.mol"))
    for index, moiety in enumerate(pubchemFP):
        result = subprocess.run(
            [str(matchmol_path), str(moiety), str(mol_file)],
            capture_output=True, text=True, check=True
            )
        if 'T' in result.stdout:
            pcfp[index] = 1
    return pcfp

def mol_to_ringsindrugs(mol_file):
    rd = np.zeros(147, dtype=int)
    src_dir = Path(__file__).resolve().parent
    matchmol_path = src_dir / "matchmol"
    ring_dir = src_dir / "RD_147"
    ringFP = sorted(ring_dir.glob("*.mol"))
    for index, moiety in enumerate(ringFP):
        result = subprocess.run(
            [str(matchmol_path), str(moiety), str(mol_file)],
            capture_output=True, text=True, check=True
            )
        if 'T' in result.stdout:
            rd[index] = 1
    return rd

def mol_to_ac(mol_file):
    ac_tmp_array = np.zeros(10)
    output_lines_ac = []
    ac_path = Path(__file__).resolve().parent / "mod_ac"
    result = subprocess.run(
        [str(ac_path), str(mol_file)],
        capture_output=True, text=True, check=True
    )
    output_ac_lines = result.stdout.splitlines()
    # print('compound_name, C.ring, C.other, N.ring, N.other, O.ring, O.other, P, S, X, #.of.Ring')
    return output_ac_lines[-1].split('\t')

def mol_to_stereo(mol_file):
    return smiles_to_stereo(mol_to_smiles(mol_file))

def mol_to_smiles(mol_file):
    mol = Chem.MolFromMolFile(mol_file, sanitize=True)
    if mol is None:
        raise ValueError("Failed to load: " + mol_file)
    return Chem.MolToSmiles(mol, canonical=True)

def smiles_to_stereo(smiles_str):
    special_char_count = {'@': 0, '\\': 0, '/': 0, 'continue_m': 0, 'continue_mm': 0, 'first_m': 0, 'first_mm': 0}
    # 遍歷字符並計數
    for char in special_char_count:
        special_char_count[char] = smiles_str.count(char)
    # 連續@ or @@ 數量
    continue_m = 0
    continue_mm = 0
    first_m = 0
    first_mm = 0
    max_continiue_m = []
    max_continiue_mm = []
    for i in range(len(smiles_str)):
        if smiles_str[i] == '@':
            if smiles_str[i+1] != '@' and smiles_str[i-1] != '@': #僅考慮@
                if first_mm == 0:
                    first_m = 1
                continue_m+=1
                if continue_mm !=0:
                    if continue_mm == 1:
                        continue_mm = 0
                        max_continiue_mm.append(continue_mm) 
                    else:    
                        max_continiue_mm.append(continue_mm)
                        continue_mm = 0            
            elif smiles_str[i+1] == '@': #僅考慮@@
                if first_m == 0:
                    first_mm = 1
                continue_mm+=1
                if continue_m != 0:
                    if continue_m == 1:
                        continue_m = 0
                        max_continiue_m.append(continue_m) 
                    else:    
                        max_continiue_m.append(continue_m)
                        continue_m = 0
    if continue_m ==1:
        max_continiue_m.append(0)
    else:
        max_continiue_m.append(continue_m)
    if continue_mm ==1:
        max_continiue_mm.append(0)
    else:
        max_continiue_mm.append(continue_mm)
    special_char_count['continue_m'] = max(max_continiue_m)
    special_char_count['continue_mm'] = max(max_continiue_mm)
    special_char_count['first_m'] = first_m
    special_char_count['first_mm'] = first_mm

    print('@, \, /, continue_@, continue_@@, first_@, first_@@')
    return list(special_char_count.values())


def smiles_to_temp_mol(smiles_str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles_str}")
    molblock = Chem.MolToMolBlock(mol)
    project_dir = Path(__file__).resolve().parent.parent
    mol_path = project_dir / "data" / "temp_smiles.mol"
    mol_path.parent.mkdir(parents=True, exist_ok=True)
    mol_path.write_text(molblock, encoding="utf-8")
    return str(mol_path)


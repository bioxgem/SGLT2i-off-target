import json
import csv
import os
import pandas as pd
import tqdm
import json
import requests


'''mapping of ligands from wwPDB to RCSB-PDB'''

def check_via_graphql(ligand_id_list):
    url = "https://data.rcsb.org/graphql"

    query_template = """
    {
      chem_comps(comp_ids: %s) {
        rcsb_id
      }
    }
    """

    valid_ids = []
    batch_size = 100
    for i in range(0, len(ligand_id_list), batch_size):
        batch = ligand_id_list[i:i+batch_size]
        formatted_ids = json.dumps(batch)
        
        payload = {"query": query_template % formatted_ids}
        res = requests.post(url, json=payload)
        
        if res.status_code == 200:
            data = res.json().get('data', {})
            if data and data['chem_comps']:
                for item in data['chem_comps']:
                    if item: # 如果 ID 存在，item 不為空
                        valid_ids.append(item['rcsb_id'])
        
        print(f"Progress: {i+len(batch)} / {len(ligand_id_list)}")
        
    return valid_ids


'''count number of pairing proteins and pairs from pdb ligands'''

def load_ligand_list(txt_path):
    if not os.path.exists(txt_path):
        print(f"{txt_path} file not found")
        return set()
        
    with open(txt_path, 'r', encoding='utf-8') as f:
        ligands = {line.strip() for line in f if line.strip()}
    return ligands

def analyze_specified_ligands(json_path, txt_path, output_csv="matched_pairs.csv"):
    # 1. read Ligand List
    target_ligands = load_ligand_list(txt_path)
    if not target_ligands:
        return

    print(f"{txt_path}, num of liands = {len(target_ligands):,}")

    # 2. read JSON Map
    print(f"read {json_path} ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        pdb_map = json.load(f)

    matched_pairs = []
    matched_proteins = set()
    found_ligands = set()
    missing_ligands = set()

    # 3. 逐一比對 Target Ligand List
    for lig_id in target_ligands:
        if lig_id in pdb_map:
            found_ligands.add(lig_id)
            # 去重：確保同一個 Ligand 對應的 UniProt ID 不重複
            unique_uniprots = set(pdb_map[lig_id])
            
            for uniprot_id in unique_uniprots:
                matched_pairs.append((lig_id, uniprot_id))
                matched_proteins.add(uniprot_id)
        else:
            missing_ligands.add(lig_id)

    # 4. 匯出 CSV 檔案
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Ligand_ID', 'UniProt_ID'])
        writer.writerows(matched_pairs)

    # 5. 輸出分析報告
    print("\n")
    print(f"num of ligands from input file  : {len(target_ligands):,}")
    print(f"num of ligands found in JSON    : {len(found_ligands):,}")
    if missing_ligands:
        print(f"num of ligands not found in JSON: {len(missing_ligands):,}")
    print(f"------------------------------------------")
    print(f"num of unique proteins          : {len(matched_proteins):,}")
    print(f"num of unique pairs             : {len(matched_pairs):,}")
    print(f"------------------------------------------")
    print(f"data saved as                   : {output_csv}")


'''mapping PDB ligands and BindingDB monomers using inchikey'''

def parse_cif_for_inchikeys(cif_path, target_pdb_ids):
    """
    get InChIKey of PDB Ligands using Ligand ID from components.cif
    """
    target_pdb_ids = set(target_pdb_ids)
    pdb_to_inchikey = {}
    
    print(f"read {cif_path}...")
    current_id = None
    
    with open(cif_path, 'r') as f:
        for line in tqdm(f, desc="Parsing CIF"):
            if line.startswith("data_"):
                current_id = line.strip().split('_')[1]
            
            if current_id in target_pdb_ids:
                if "_chem_comp.pdbx_model_Cartn_x_ideal" in line:
                    pass
                if "_item_details.value" in line: 
                    pass
                
                if "_chem_comp.pdbx_HL_standard_InChIKey" in line or "InChIKey" in line:
                    parts = line.split()
                    for p in parts:
                        if len(p) == 27 and p[14] == '-' and p[25] == '-':
                            pdb_to_inchikey[current_id] = p
    return pdb_to_inchikey

def inchikey_mapping():
    '''components.cif can be downloaded from https://www.wwpdb.org/data/ccd'''
    cif_file = "../PDB_ligand/components.cif"

    bdb_json = "../BindingDB_data/bdb_id_to_inchikey.json" 
    output_csv = "./data_processing/pdb_to_bdb_mapping.csv"
    
    my_pdb_ids = pd.read_csv("./data_processing/pdb_predicted_inhibitors.txt", sep="\t")["PDB_ID"].tolist()

    print(f"read {bdb_json}...")
    with open(bdb_json, 'r') as f:
        bdb_data = json.load(f)
    
    inchikey_to_bdb = {}
    for bdb_id, ikey in bdb_data.items():
        if ikey not in inchikey_to_bdb:
            inchikey_to_bdb[ikey] = []
        inchikey_to_bdb[ikey].append(str(bdb_id))

    pdb_to_ikey = parse_cif_for_inchikeys(cif_file, my_pdb_ids)

    results = []
    for p_id in my_pdb_ids:
        ikey = pdb_to_ikey.get(p_id, "Not Found in CIF")
        b_ids = "Not Found in BDB"
        
        if ikey in inchikey_to_bdb:
            b_ids = ";".join(inchikey_to_bdb[ikey])
            
        results.append({
            "PDB_Ligand_ID": p_id,
            "bdb_id": b_ids,
            "InChIKey": ikey
        })

    df_final = pd.DataFrame(results)
    df_final.to_csv(output_csv, index=False)
    print(f"data saved to {output_csv}")

import pandas as pd
import pickle as pkl
import networkx as nx
import os

"""
This script processes the codes by arranging them into two data structures:
1. A dictionary mapping from code to node name in the graph
2. A networkx graph
"""

path = '/work/frink/mcinerney.de/icd_codes'
icd9_file = os.path.join(path, 'diagnoses/icd9.txt')
icd10_file = os.path.join(path, 'diagnoses/icd10.txt')
icd9to10_file = os.path.join(path, 'diagnoses/icd9to10.csv')
icd10to9_file = os.path.join(path, 'diagnoses/icd10to9.csv')
code_mapping_file = os.path.join(path, 'code_mapping.pkl')
code_graph_file = os.path.join(path, 'code_graph.pkl')

def process_file(filename1, filename2, split_at, encoding='utf-8'):
    with open(filename1, 'r', encoding=encoding) as file:
        with open(filename2, 'w') as file2:
            for i,line in enumerate(file):
                file2.write(line[:split_at].strip()+'\t'+line[split_at:].strip()+'\n')

def create_node_and_parents(G, code_mapping, code, description=None):
    curr_node = str(('ICD10', code))
    G.add_node(curr_node, description=description)
    # TODO: is this a bug below? should fix, but keeping the same now until I can do tests.
    # I think this should not be a problem if you are using radiology codes bc it is sort of corrected in radiology_codes.py
    code_mapping['curr_node'] = curr_node
    for l in range(len(code)-1,-1,-1):
        parent = str(('ICD10', code[:l]))
        if parent not in G.nodes:
            G.add_node(parent, description=None)
        G.add_edge(parent, curr_node)
        curr_node = parent

def main():
    process_file(icd9_file, icd9_file+'_processed.csv', 6, encoding="ISO-8859-1")
    icd9 = pd.read_csv(icd9_file+'_processed.csv', names=['code', 'description'], delimiter='\t')
    process_file(icd10_file, icd10_file+'_processed.csv', 8)
    icd10 = pd.read_csv(icd10_file+'_processed.csv', names=['code', 'description'], delimiter='\t')
    icd9to10 = pd.read_csv(icd9to10_file)
    #icd10to9 = pd.read_csv(icd10to9_file)
    G = nx.DiGraph()
    # make icd10 graph and start code mapping
    code_mapping = {}
    for i,row in icd10.iterrows():
        create_node_and_parents(G, code_mapping, row.code, description=row.description)
    # add icd9 entries to code mapping
    for i,row in icd9.iterrows():
        icd10code = icd9to10.icd10cm[icd9to10.icd9cm == row.code].iloc[0]
        if icd10code == 'NoDx':
            print("WARNING: %s has no corresponding icd10 code, it is not being included in the mapping" % row.code)
            continue
        code_mapping[str(('ICD9', row.code))] = str(('ICD10', icd10code))
        if str(('ICD10', icd10code)) not in G.nodes:
            create_node_and_parents(G, code_mapping, icd10code)
    with open(code_mapping_file, 'wb') as f:
        pkl.dump(code_mapping, f)
    with open(code_graph_file, 'wb') as f:
        pkl.dump(G, f)
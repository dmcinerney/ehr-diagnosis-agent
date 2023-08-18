import pandas as pd
import pickle as pkl
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_postorder_nodes
from networkx.classes.function import set_node_attributes, subgraph
from tqdm import tqdm
import os

path = '/work/frink/mcinerney.de/icd_codes'
outline_file = 'icd_codes/radiology_codes_outline.txt'
outline_type = 'ICD9'
original_code_mapping_file = os.path.join(path, 'code_mapping.pkl')
original_code_graph_file = os.path.join(path, 'code_graph.pkl')
only_leaves = True
expanded = True
if expanded:
    new_code_mapping_file = os.path.join(path, 'code_mapping_radiology_expanded.pkl')
    new_code_graph_file = os.path.join(path, 'code_graph_radiology_expanded.pkl')
else:
    new_code_mapping_file = os.path.join(path, 'code_mapping_radiology.pkl')
    new_code_graph_file = os.path.join(path, 'code_graph_radiology.pkl')

def create_graph(filename):
    G = nx.DiGraph()
    prev_tab_level = None
    start_node = '<start_node>'
    current_nodes_at_each_depth = []
    with open(filename, 'r') as file:
        for i,line in enumerate(file):
            if len(line.strip()) == 0: continue
            num_spaces = len(line) - len(line.lstrip(' '))
            new_tab_level = num_spaces//4
            tab_difference = new_tab_level - prev_tab_level if prev_tab_level is not None else 0
            prev_tab_level = new_tab_level
            # print(i+1, new_tab_level, tab_difference)
            if tab_difference > 1:
                raise Exception
            if len(current_nodes_at_each_depth) < new_tab_level + 1:
                current_nodes_at_each_depth.append(None)
            linesplit = line.strip().split()
            indices = [i for i,token in enumerate(linesplit) if any(char.isdigit() for char in token)]
            if len(indices) > 0:
                nodename = linesplit[indices[0]]
                del linesplit[indices[0]]
                description = ' '.join(linesplit)
            else:
                nodename = line.strip()
                #description = nodename
                description = None
            if nodename in G.nodes:
                print("NEED TO DEBUG OUTLINE! MORE THAN ONE OCCURANCE OF THE SAME NODE")
                import pdb; pdb.set_trace()
                raise Exception
            G.add_node(nodename, description=description)
            current_nodes_at_each_depth[new_tab_level] = nodename
            if new_tab_level == 0:
                G.add_edge(start_node, nodename)
            else:
                G.add_edge(current_nodes_at_each_depth[new_tab_level-1], nodename)
    return G

def splitints(string):
    letters = ''
    numbers = ''
    for char in string:
        if char.isdigit():
            numbers += char
        else:
            letters += char
    return letters, numbers

# really hacky
def getids(nodename):
    if not any(char.isdigit() for char in nodename):
        raise Exception
    nodename = nodename.replace('.','')
    if '-' in nodename:
        start, stop = nodename.split('-')
        startl, startn = splitints(start)
        stopl, stopn = splitints(stop)
        if startl != stopl or (startl+startn) != start or (stopl+stopn) != stop:
            raise Exception
        return [startl+str(i)[1:] for i in range(int('1'+startn), int('1'+stopn)+1)]
    else:
        return [nodename]

def get_codes_under_category(n, df, original_G, original_code_mapping, outline_type=None):
    ids = getids(n)
    if outline_type is None:
        # auto decide if icd 9 or 10
        codes9 = []
        codes10 = []
        for i in ids:
            codes9 += list(df[0][df[0].str.startswith(str(('ICD9', i))[:-2])])
            codes10 += list(df[0][df[0].str.startswith(str(('ICD10', i))[:-2])])
        if len(codes10) > 0 and len(codes10) > len(codes9):
            type = 'ICD10'
        elif len(codes9) > 0:
            type = 'ICD9'
        else:
            return []
    else:
        type = outline_type
        if type == 'ICD9':
            codes9 = []
            for i in ids:
                codes9 += list(df[0][df[0].str.startswith(str(('ICD9', i))[:-2])])
    if type == 'ICD10':
        codes = []
        for i in ids:
            codes += [code for code in original_G.nodes if code.startswith(str(('ICD10', i))[:-2])]
    elif type == 'ICD9':
        codes = [original_code_mapping[code] for code in codes9]
    #if len(codes) == 0: import pdb; pdb.set_trace()
    return codes

def non_terminal_leaf_node(G, n):
    print("leaf node: %s has no ICD code children! Deleting nodes on this branch until a non-leaf node is hit!" % n)
    curr_node = n
    while G.out_degree(curr_node) == 0:
        parents = list(G.predecessors(curr_node))
        G.remove_node(curr_node)
        if len(parents) == 1:
            curr_node = parents[0]
        elif len(parents) == 0: break
        elif len(parents) > 1: raise Exception

def create_icd_mapping_and_modify_graph(G, original_G, original_code_mapping, only_leaves=False, expanded=False, outline_type=None):
    new_code_mapping = {}
    df = pd.DataFrame(list(original_code_mapping.keys()))
    postorder = list(dfs_postorder_nodes(G, source='<start_node>'))
    for n in tqdm(postorder, total=len(postorder)):
        if only_leaves and G.out_degree(n) != 0:
            continue
        if not any(char.isdigit() for char in n):
            raise Exception
        codes = get_codes_under_category(n, df, original_G, original_code_mapping, outline_type=outline_type)
        if G.out_degree(n) == 0 and len(codes) == 0:
            non_terminal_leaf_node(G, n)
            continue
        codes = sorted(codes, key=lambda x: len(x))
        node_subset = set()
        for code in codes:
            if code in node_subset:
                continue
            if code in G.nodes:
                print("conflict during %s: %s -> %s already exists!" % (n, next(iter(G.predecessors(code))), code))
                continue
            node_subset.add(code)
            node_subset = node_subset.union(nx.descendants(original_G, code))
        if G.out_degree(n) == 0 and len(node_subset) == 0:
            non_terminal_leaf_node(G, n)
            continue
        for node in node_subset:
            if expanded:
                new_code_mapping[node] = node
                pred = next(original_G.predecessors(node))
                if pred in node_subset:
                    G.add_edge(pred, node)
                else:
                    G.add_edge(n, node)
                G.nodes[node]['description'] = original_G.nodes[node]['description']
            else:
                new_code_mapping[node] = n
                if 'codes' not in G.nodes[n].keys():
                    G.nodes[n]['codes'] = set()
                G.nodes[n]['codes'].add(node)
        """
        for code in codes:
            code = original_code_mapping[code]
            new_code_mapping[code] = n
            node_codes[n]['codes'].add(code)
            for descendant in nx.descendants(original_G, code):
                new_code_mapping[descendant] = n
                node_codes[n]['codes'].add(descendant)
        """
    for k,v in original_code_mapping.items():
        if v not in new_code_mapping.keys():
            new_code_mapping[k] = None
        else:
            new_code_mapping[k] = new_code_mapping[v]
    return new_code_mapping

def depth(G, n):
    count = 0
    preds = list(G.predecessors(n))
    while len(preds) > 0:
        n = next(iter(preds))
        preds = list(G.predecessors(n))
        count += 1
    return count

def main():
    with open(original_code_mapping_file, 'rb') as f:
        original_code_mapping = pkl.load(f)
    with open(original_code_graph_file, 'rb') as f:
        original_G = pkl.load(f)
    G = create_graph(outline_file)
    print([n for n in original_G.nodes if original_G.in_degree(n) > 1])
    print([n for n in G.nodes if G.in_degree(n) > 1])
    new_code_mapping = create_icd_mapping_and_modify_graph(G, original_G, original_code_mapping, expanded=expanded, outline_type=outline_type, only_leaves=only_leaves)
    print([n for n in G.nodes if G.in_degree(n) > 1])
    with open(new_code_mapping_file, 'wb') as f:
        pkl.dump(new_code_mapping, f)
    with open(new_code_graph_file, 'wb') as f:
        pkl.dump(G, f)
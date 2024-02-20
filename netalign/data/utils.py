import json
import os
from itertools import chain
from typing import List

import networkx as nx
import numpy as np
import torch
from networkx.readwrite import json_graph
from scipy.io import loadmat
from torch_geometric.utils import (add_random_edge, contains_self_loops,
                                   dropout_edge, is_undirected,
                                   remove_self_loops, shuffle_node,
                                   to_undirected)
from torch_geometric.utils.convert import from_networkx


def network_info(G, n=None):
    """Print short summary of information for the graph G or the node n.

    Args:
        G: A NetworkX graph.
        n:  A node in the graph G
    """
    info='' # Append this all to a string.
    if n is None:
        info+="Name: %s\n"%G.name
        type_name = [type(G).__name__]
        info+="Type: %s\n"%",".join(type_name)
        info+="Number of nodes: %d\n"%G.number_of_nodes()
        info+="Number of edges: %d\n"%G.number_of_edges()
        nnodes=G.number_of_nodes()
        if len(G) > 0:
            if G.is_directed():
                info+="Average in degree: %8.4f\n"%\
                    (sum(G.in_degree().values())/float(nnodes))
                info+="Average out degree: %8.4f"%\
                    (sum(G.out_degree().values())/float(nnodes))
            else:
                degrees = dict(G.degree())
                s=sum(degrees.values())
                info+="Average degree: %8.4f"%\
                    (float(s)/float(nnodes))

    else:
        if n not in G:
            raise nx.NetworkXError("node %s not in graph"%(n,))
        info+="Node % s has the following properties:\n"%n
        info+="Degree: %d\n"%G.degree(n)
        info+="Neighbors: "
        info+=' '.join(str(nbr) for nbr in G.neighbors(n))

    return info


def edgelist_to_graphsage(dir, seed=42):
    np.random.seed(seed)
    edgelist_path = dir + "/edgelist/edgelist"

    # Check if the edge list is weighted or unweighted
    with open(edgelist_path, 'r') as file:
        first_line = file.readline().strip().split()
        is_weighted = len(first_line) > 2

    if is_weighted:
        G = nx.read_weighted_edgelist(edgelist_path)
    else:
        G = nx.read_edgelist(edgelist_path)

    print(network_info(G))

    id2idx = {}
    for i, node in enumerate(G.nodes()):
        id2idx[str(node)] = i

    res = json_graph.node_link_data(G)
    res['nodes'] = [{'id': node['id']} for node in res['nodes']]

    if is_weighted:
        res['links'] = [{'source': link['source'],
                         'target': link['target'],
                         'weight': link['weight']}
                         for link in res['links']]
    else:
        res['links'] = [{'source': link['source'],
                         'target': link['target']}
                         for link in res['links']] 

    if not os.path.exists(dir + "/graphsage/"):
        os.makedirs(dir + "/graphsage/")

    with open(dir + "/graphsage/" + "G.json", 'w') as outfile:
        json.dump(res, outfile)
    with open(dir + "/graphsage/" + "id2idx.json", 'w') as outfile:
        json.dump(id2idx, outfile)


def print_graph_stats(G):
    print('# of nodes: %d, # of edges: %d' % (G.number_of_nodes(),
                                              G.number_of_edges()))
    
def construct_adjacency(G, id2idx):
    adjacency = np.zeros((len(G.nodes()), len(G.nodes())))
    for src_id, trg_id in G.edges():
        adjacency[id2idx[src_id], id2idx[trg_id]] = 1
        adjacency[id2idx[trg_id], id2idx[src_id]] = 1
    return adjacency


def build_degrees(G, id2idx):
    degrees = np.zeros(len(G.nodes()))
    for node in G.nodes():
        deg = G.degree(node)
        degrees[id2idx[node]] = deg
    return degrees


def build_clustering(G, id2idx):
    cluster = nx.clustering(G)
    # convert clustering from dict with keys are ids to array index-based
    clustering = [0] * len(G.nodes())
    for id, val in cluster.items():
        clustering[id2idx[id]] = val
    return clustering


def get_H(path, source_dataset, target_dataset):
    
    if path is None:    
        H = np.ones((len(target_dataset.G.nodes()), len(source_dataset.G.nodes())))
        H = H*(1/len(source_dataset.G.nodes()))
        return H
    else:    
        if not os.path.exists(path):
            raise Exception("Path '{}' is not exist".format(path))
        dict_H = loadmat(path)
        H = dict_H['H']
        return H


def get_edges(G, id2idx):
    edges1 = [(id2idx[n1], id2idx[n2]) for n1, n2 in G.edges()]
    edges2 = [(id2idx[n2], id2idx[n1]) for n1, n2 in G.edges()]
    
    edges = edges1 + edges2
    edges = np.array(edges)
    return edges


def load_gt(path, id2idx_src, id2idx_trg, format='matrix', convert=False):    
    conversion_src = type(list(id2idx_src.keys())[0])
    conversion_trg = type(list(id2idx_trg.keys())[0])
    if format == 'matrix':
        gt = np.zeros((len(id2idx_src.keys()), len(id2idx_trg.keys())))
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()                
                gt[id2idx_src[conversion_src(src)], id2idx_trg[conversion_trg(trg)]] = 1
        return gt
    else:
        gt = {}
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()
                if convert:
                    gt[id2idx_src[conversion_src(src)]] = id2idx_trg[conversion_trg(trg)]
                else:
                    gt[conversion_src(src)] = conversion_trg(trg)
        return gt



def compute_node_metric(G, metric: str):
    if metric == 'degree':
        return dict(G.degree())
    if metric == 'pagerank':
        return nx.pagerank(G)
    raise ValueError(f"Invalid node metric name {metric}.")
    
    
def compute_edge_metric(G, metric):
    if metric == 'betwenneess':
        return nx.edge_betweenness_centrality(G, weight='weight')
    raise ValueError(f"Invalid edge metric name {metric}.")


def filter_files_with_substrings(directory, substring_list):
    matching_files = []

    # List all files in the directory
    files = os.listdir(directory)

    for file in files:
        # Check if the file name contains all substrings in the substring list
        if all(substring in file for substring in substring_list):
            matching_files.append(file)

    return matching_files
    
    
def generate_synth_clone(source_pyg, p_rm=0.0, p_add=0.0):
    # Clone source pyg graph.
    target_pyg = source_pyg.clone() 
    
    # Check if undirected.
    if is_undirected(source_pyg.edge_index):
        force_undirected = True      
    
    # Remove edges with probability.
    target_pyg.edge_index, edge_mask = dropout_edge(target_pyg.edge_index,
                                                    p=p_rm,
                                                    force_undirected=force_undirected)
    target_pyg.edge_attr = target_pyg.edge_attr[edge_mask]
    
    # Add edges with probability.
    target_pyg.edge_index, added_edges = add_random_edge(target_pyg.edge_index,
                                                            p=p_add,
                                                            force_undirected=force_undirected)
    
    # Sample edge attributes to assign them to the new added edge index.
    # Sampling consent us to obtain not trivial attributes values
    # easily recognisable from a GCN.
    num_new_edges = added_edges.size(1)
    sample_indices = torch.randperm(target_pyg.edge_attr.size(0))[:num_new_edges]
    old_attr = target_pyg.edge_attr
    new_attr = target_pyg.edge_attr[sample_indices]
    target_pyg.edge_attr = torch.cat((old_attr, new_attr), dim=0)

    # Shuffle nodes.
    target_pyg.x, node_perm = shuffle_node(target_pyg.x)
    
    # Make it undirected and remove self loops.
    if force_undirected is True and not is_undirected(target_pyg.edge_index):
        target_pyg.edge_index, target_pyg.edge_attr = to_undirected(target_pyg.edge_index, target_pyg.edge_attr)
    if contains_self_loops(target_pyg.edge_index):
        target_pyg.edge_index, target_pyg.edge_attr = remove_self_loops(target_pyg.edge_index, target_pyg.edge_attr)
        
    # Build the groundtruth alignment matrix.
    gt_perm_mat = torch.zeros((source_pyg.num_nodes, target_pyg.num_nodes), dtype=torch.float)
    for s, t in enumerate(node_perm):   # `node_perm` contains the order of original nodes after shuffling.
        gt_perm_mat[s, t] = 1
        
    return target_pyg, gt_perm_mat


def to_pyg_graph(G,
                id2idx: dict,
                node_feats: torch.Tensor=None,
                edge_feats: torch.Tensor=None,
                node_metrics: List[str]=[],
                edge_metrics: List[str]=[]):
    
    # Assign existing features.
    if node_feats is not None:
        for n in G.nodes:
            G.nodes[n]['features'] = torch.Tensor(node_feats[id2idx[n]])
    
    if edge_feats is not None:
        for e, f in zip(G.edges, edge_feats):
            G.edges[e]['features'] = torch.Tensor(f)
            
    # Explicit edge weight to 1 if unweighted.
    if not nx.is_weighted(G):
        nx.set_edge_attributes(G, 1, name='weight')
            
    # Generate new node/edge features using local metrics.
    if len(node_metrics) > 0:
        for metric in node_metrics:
            feats = compute_node_metric(G, metric)
            nx.set_node_attributes(G, feats, name=metric) 
            
    if len(edge_metrics) > 0:
        for metric in edge_metrics:
            feats = compute_edge_metric(G, metric)
            nx.set_edge_attributes(G, feats, name=metric)
    
    
    # Get the list of node/edge attribute names.
    node_attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True))) - set(['val', 'test'])
    edge_attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
    
    # Convert to pyg
    pyg_graph = from_networkx(G,
                              group_node_attrs=node_attrs_list,
                              group_edge_attrs=edge_attrs_list)
    
    
    # Set default node importance to 1 if not present yet
    if 'x_importance' not in pyg_graph:
        pyg_graph.x_importance = torch.ones([pyg_graph.num_nodes, 1])
        
    # Make it undirected and remove self loops
    if not nx.is_directed(G) and not is_undirected(pyg_graph.edge_index):
        pyg_graph.edge_index, pyg_graph.edge_attr = to_undirected(pyg_graph.edge_index, pyg_graph.edge_attr)
    if contains_self_loops(pyg_graph.edge_index):
        pyg_graph.edge_index, pyg_graph.edge_attr = remove_self_loops(pyg_graph.edge_index, pyg_graph.edge_attr)
                
    
    return pyg_graph


def train_test_split(matrix, split_ratio=0.2):
    """
    Given a matrix of shape (N,M) representing the alignments
    between the nodes of a source network with the nodes of
    a target network, split those alignments in two set using
    the `split_ratio` and return them as dictionaries.
    """
    
    # Get alignment indices
    gt_indices = torch.argwhere(matrix == 1)
    num_alignments = gt_indices.shape[0]
    assert gt_indices.shape == torch.Size([num_alignments, 2]) 
    
    # Shuffling
    shuffled_idx = torch.randperm(num_alignments)
    gt_indices = gt_indices[shuffled_idx]
    
    # Split indices
    split_size = int(num_alignments * split_ratio)
    split_sizes = [split_size, num_alignments - split_size]
    train, test = torch.split(gt_indices, split_sizes, dim=0)

    # Generate dictionaries
    train_dict = {k.item(): v.item() for k, v in train}
    test_dict = {k.item(): v.item() for k, v in test}
    
    return train_dict, test_dict
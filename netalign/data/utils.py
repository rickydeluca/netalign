import json
import os
import random
from itertools import chain

import networkx as nx
import numpy as np
import torch
from networkx.readwrite import json_graph
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import (add_random_edge, contains_self_loops,
                                   dropout_edge, is_undirected,
                                   remove_self_loops)
from torch_geometric.utils.convert import from_networkx


def network_info(G, n=None):
    """Print short summary of information for the graph G or the node n.

    Args:
        G: A NetworkX graph.
        n:  A node in the graph G
    """
    info='' # Append this all to a string
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


def edgelist_to_networkx(dir, seed=42):
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
    return G


def edgelist_to_pyg(dir, seed=42):
    # Load graph in NetworkX format
    G = edgelist_to_networkx(dir, seed)

    # Explicit edge weight to 1 if unweighted
    if not nx.is_weighted(G):
        nx.set_edge_attributes(G, 1, name='weight')

    # Get list of node and edge attributes
    node_attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))
    edge_attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))

    # Convert to PyG
    group_node_attrs = node_attrs_list if len(node_attrs_list) > 0 else None
    group_edge_attrs = edge_attrs_list if len(edge_attrs_list) > 0 else None

    pyg_graph = from_networkx(G,
                              group_node_attrs=group_node_attrs,
                              group_edge_attrs=group_edge_attrs)
    
    pyg_graph.num_nodes = G.number_of_nodes()

    # Make it undirected
    pyg_graph = ToUndirected()(pyg_graph)

    # Remove self loops
    if contains_self_loops(pyg_graph.edge_index):
        pyg_graph.edge_index, pyg_graph.edge_attr = remove_self_loops(pyg_graph.edge_index,
                                                                      pyg_graph.edge_attr)

    return pyg_graph


def generate_random_synth_clone(pyg_source, p_rm=0.0, p_add=0.0):
    # Clone graph
    pyg_target = pyg_source.clone()

    if is_undirected(pyg_source.edge_index):
        force_undirected = True

    # Permute graph
    unique_nodes = torch.unique(pyg_target.edge_index)
    permuted_nodes = torch.randperm(unique_nodes.size(0))
    node_mapping = dict(zip(unique_nodes.numpy(), permuted_nodes.numpy()))

    permuted_edge_index = torch.tensor([
        [node_mapping[pyg_target.edge_index[0, i].item()] for i in range(pyg_target.edge_index.size(1))],
        [node_mapping[pyg_target.edge_index[1, i].item()] for i in range(pyg_target.edge_index.size(1))]
    ], dtype=torch.long)

    pyg_target.edge_index = permuted_edge_index

    # Remove edges with probability
    pyg_target.edge_index, edge_mask = dropout_edge(pyg_target.edge_index,
                                                    p=p_rm,
                                                    force_undirected=force_undirected)
    pyg_target.edge_attr = pyg_target.edge_attr[edge_mask]

    # Add edges with probability
    pyg_target.edge_index, added_edges = add_random_edge(pyg_target.edge_index,
                                                         p=p_add,
                                                         force_undirected=force_undirected)
    
    # Sample edge attributes for new edges to avoid
    # creating trivial edge attributes
    num_new_edges = added_edges.size(1)
    sample_indices = torch.randperm(pyg_target.edge_attr.size(0))[:num_new_edges]
    old_attr = pyg_target.edge_attr
    new_attr = pyg_target.edge_attr[sample_indices]
    pyg_target.edge_attr = torch.cat((old_attr, new_attr), dim=0)

    # Remove any evantual self loop
    if contains_self_loops(pyg_target.edge_index):
        pyg_target.edge_index, pyg_target.edge_attr = remove_self_loops(pyg_target.edge_index,
                                                                        pyg_target.edge_attr)

    return pyg_target, node_mapping
    

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


def shuffle_and_split(dictionary, split_ratio, seed=42):
    """
    Shuffle the items in the dictionary and split it based on the given ratio.

    Args:
        dictionary (dict): The input dictionary to shuffle and split.
        split_ratio (float): The ratio at which to split the dictionary.

    Returns:
        dict, dict: Two dictionaries representing the split datasets.
    """
    random.seed(seed)    

    # Convert dictionary items to a list of tuples
    items = list(dictionary.items())

    # Shuffle the items
    random.shuffle(items)

    # Calculate the split index
    split_index = int(len(items) * split_ratio)

    # Split the items into two lists
    split_items_1 = items[:split_index]
    split_items_2 = items[split_index:]

    # Convert the split lists back to dictionaries
    split_dict_1 = {k: torch.tensor(v).item() for k, v in split_items_1}
    split_dict_2 = {k: torch.tensor(v).item() for k, v in split_items_2}

    return split_dict_1, split_dict_2
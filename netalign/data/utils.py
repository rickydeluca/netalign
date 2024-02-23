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
    """
    Read the edgelist file in the `dir` path, check if it
    is weighted and return the corresponding NetworkX graph.
    """

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
        nx.set_edge_attributes(G, float(1), name='weight') # Explicit weight value

    print(network_info(G))

    return G


def get_node_attribute_names(G):
    """
    Return the list with the name of node attributes
    of the NetworkX graph `G`.

    If `G` has no node attributes return `None`.
    """
    attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))

    if len(attrs_list) > 0:
        return attrs_list
    else:
        return None


def get_edge_attribute_names(G):
    """
    Return the list with the name of edge attributes
    of the NetworkX graph `G`.

    If `G` has no edge attributes return `None`.
    """
    attrs_list = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
    
    if len(attrs_list) > 0:
        return attrs_list
    else:
        return None


def edgelist_to_pyg(dir, seed=42):
    """
    Converts a graph from an edge list in the specified directory to a PyTorch Geometric (PyG) graph.

    Args:
        dir (str): The directory containing the edge list file.
        seed (int, optional): Seed for random number generation. Default is 42.

    Returns:
        torch_geometric.data.Data: PyTorch Geometric graph object representing the loaded graph.
    """
    # Load graph in NetworkX format
    G = edgelist_to_networkx(dir, seed)

    # Get list of node and edge attributes
    node_attrs_list = get_node_attribute_names(G)
    edge_attrs_list = get_edge_attribute_names(G)

    # Convert to PyG
    pyg_graph = from_networkx(
        G,
        group_node_attrs=node_attrs_list,
        group_edge_attrs=edge_attrs_list
    )

    pyg_graph.num_nodes = G.number_of_nodes()

    # Remove self loops
    if contains_self_loops(pyg_graph.edge_index):
        pyg_graph.edge_index, pyg_graph.edge_attr = remove_self_loops(pyg_graph.edge_index,
                                                                      pyg_graph.edge_attr)

    return pyg_graph


def permute_graph(pyg_source):
    """
    Permute the indices of the pytorch geometric 
    Data object `pyg_source` and return the permuted
    copy.
    """
    # Clone graph
    pyg_target = pyg_source.clone()

    # Permute graph
    num_nodes = pyg_target.num_nodes
    edge_index = pyg_target.edge_index
    # edge_attr = pyg_target.edge_attr
    x = pyg_target.x

    perm_indices = torch.randperm(num_nodes)
    perm_edge_index = perm_indices[edge_index]
    # perm_edge_attr = edge_attr[perm_edge_index]
    perm_x = x[perm_indices] if x is not None else None
    
    pyg_target.edge_index = perm_edge_index
    # pyg_target.edge_attr = perm_edge_attr
    pyg_target.x = perm_x

    # Get groundtruth mapping
    mapping = {k: v.item() for k, v in enumerate(perm_indices)}

    return pyg_target, mapping


def sample_edge_attrs(edge_attr, num_new_attrs=0):
    """
    Sample `num_new_attrs` with the same size L of the attributes
    in `edge_attr` and value between the minimum and the maximum
    value in the `edge_attrs` tensor.

    Returns:
        torch.Tensor:   The sample new attributes of shape (num_new_attrs, L)
    """
    # Calculate min and max values along each column
    min_values, _ = torch.min(edge_attr, dim=1, keepdim=True)
    max_values, _ = torch.max(edge_attr, dim=1, keepdim=True)

    # Create a new tensor with values sampled between min and max
    sampled_attrs = torch.rand_like(edge_attr)
    sampled_attrs = sampled_attrs * (max_values - min_values) + min_values

    return sampled_attrs[:num_new_attrs, :]


def remove_random_edges(pyg_graph, p=0.0):
    """
    Drop random edges from `pyg_graph`. The probability
    of one edge to be removed is given by `p`.
    Drop also the attributes corresponding to the dropped edges.

    Returns:
        torch_geometric.data.Data:  The pyg graph with dropped edges.
    """

    edge_index = pyg_graph.edge_index
    edge_attr = pyg_graph.edge_attr

    # Check if undirected
    if is_undirected(edge_index, edge_attr):
        force_undirected = True
    else:
        force_undirected = False

    # Remove edges
    new_edge_index, edge_mask = dropout_edge(edge_index, p,
                                             force_undirected=force_undirected)
    new_edge_attr = edge_attr[edge_mask]

    # Return new graph
    pyg_graph.edge_index = new_edge_index
    pyg_graph.edge_attr = new_edge_attr

    return pyg_graph


def add_random_edges(pyg_graph, p=0.0):
    """
    Add random edges to `pyg_graph`. The percentage of new edges 
    is given by `p` and it is computed on the basis of the already
    existing edges.

    Also, generate a new node attribute for each new added edge.
    The new node attribute is sampled between the min and max 
    actual attribute values. If they are all the same a new attribute
    with the same value of those already present is sampled.

    Returns:
        torch_geometric.data.Data:  The pyg graph with new added edges.
    """

    edge_index = pyg_graph.edge_index
    edge_attr = pyg_graph.edge_attr

    # Check if undirected
    if is_undirected(edge_index, edge_attr):
        force_undirected = True
    else:
        force_undirected = False

    # Add edges
    new_edge_index, added_edges = add_random_edge(edge_index, p,
                                                  force_undirected=force_undirected)
    
    # Sample edge attributes for the new added edges
    sampled_edge_attr = sample_edge_attrs(edge_attr, num_new_attrs=added_edges.size(1))

    # Concat new samples attributes to the original attributes
    new_edge_attr = torch.cat((edge_attr, sampled_edge_attr), dim=0)

    # Return the new graph
    pyg_graph.edge_index = new_edge_index
    pyg_graph.edge_attr = new_edge_attr

    return pyg_graph
    

def generate_target_graph(pyg_source, p_rm=0.0, p_add=0.0):
    """
    Generate the permuted and noised target graph obtained from
    the pytorch geometric Data object `pyg_source`.
    
    Args:
        pyg_source (torch_geometric.data.Data):     The pytorch geometric input graph.
        p_rm (float):                               The probability of dropping an existing edge.
        p_add (float):                              The probabilty of adding a new edge.

    Returns:
        pyg_target (torch_geometric.data.Data):     The radomly permuted and noised target graph.
        mapping (dict):                             The groundtruth mapping of node indices.
    """
    # Get permuted clone
    pyg_target, mapping = permute_graph(pyg_source)

    # Remove and/or add edges with probability
    pyg_target = remove_random_edges(pyg_target, p=p_rm)
    pyg_target = add_random_edges(pyg_target, p=p_add)

    # Remove any evantual self loop
    if contains_self_loops(pyg_target.edge_index):
        pyg_target.edge_index, pyg_target.edge_attr = remove_self_loops(pyg_target.edge_index,
                                                                        pyg_target.edge_attr)

    return pyg_target, mapping
    

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
    split_dict_1 = {k: v for k, v in split_items_1}
    split_dict_2 = {k: v for k, v in split_items_2}

    return split_dict_1, split_dict_2
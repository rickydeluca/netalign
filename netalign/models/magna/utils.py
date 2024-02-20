import networkx as nx
from torch_geometric.utils import to_networkx, is_undirected


def read_align_dict(edgelist_path, sep=" ", reverse=False):
    edgelist_path += '_final_alignment.txt'
    align_dict = {}
    weighted = False

    with open(edgelist_path, 'r') as ef:
        for line in ef:
            nodes = line.strip().split(sep)
            
            if len(nodes) == 2:
                source, target = nodes

                if reverse:
                    align_dict[target] = source
                else:
                    align_dict[source] = target

            if len(nodes) == 3:
                weighted = True
                source, target, weight = nodes

                if reverse:
                    align_dict[target] = (source, weight)
                else:
                    align_dict[source] = (target, weight)

    return align_dict, weighted


def pyg_to_edgelist(pyg_graph, outfile):
    """
    Converts a PyTorch Geometric Data object to an edgelist and saves it to a file.
    The function checks if the graph is undirected and/or weighted, converts it to 
    an edgelist (with weights if necessary), and saves it in the specified outfile path.

    Args:
        pyg_graph (torch_geometric.data.Data): The PyTorch Geometric Data object representing the graph.
        outfile (str): The path to the file where the edgelist will be saved.

    Returns:
        None
    """

    # Check if weighted and/or undirected
    has_edge_weights = 'edge_attr' in pyg_graph
    undirected = is_undirected(pyg_graph.edge_index)

    # Convert to NetworkX
    nx_graph = to_networkx(pyg_graph,
                           node_attrs=['x'],
                           edge_attrs=['edge_attr'],
                           to_undirected=undirected)

    # Create edgelist
    edgelist = []
    for edge in nx_graph.edges(data=True):
        source, target, edge_data = edge
        if has_edge_weights:
            weight = edge_data['edge_attr'][0]
            edgelist.append((source, target, weight))
        else:
            edgelist.append((source, target))

    # Save the edgelist to the outfile
    with open(outfile, 'w') as f:
        for edge in edgelist:
            if has_edge_weights:
                f.write(f"{edge[0]} {edge[1]} {edge[2]}\n")
            else:
                f.write(f"{edge[0]} {edge[1]}\n")
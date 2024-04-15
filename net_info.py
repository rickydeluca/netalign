import networkx as nx
import matplotlib.pyplot as plt

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


def edgelist_to_networkx(edgelist_path, verbose=False):
    """
    Read the edgelist file in the `dir` path, check if it
    is weighted and return the corresponding NetworkX graph.
    """

    # Check if the edge list is weighted or unweighted
    with open(edgelist_path, 'r') as file:
        first_line = file.readline().strip().split()
        is_weighted = len(first_line) > 2

    if is_weighted:
        G = nx.read_weighted_edgelist(edgelist_path)
    else:
        G = nx.read_edgelist(edgelist_path)
        nx.set_edge_attributes(G, float(1), name='weight') # Explicit weight value
    
    if verbose:
        print(network_info(G))

    return G


def plot_degree_distribution(G, name='EDI3'):
    # Get the degree distribution
    degree_count = nx.degree_histogram(G)

    # Plot degree distribution
    plt.bar(range(len(degree_count)), degree_count, width=0.8, color='b')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title(f'{name} Degree Distribution')
    plt.show()


if __name__ == '__main__':
    G = edgelist_to_networkx('data/ppi/ppi.txt', verbose=True)
    plot_degree_distribution(G, name='PPI')
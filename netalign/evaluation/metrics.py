import networkx as nx
import numpy as np
import torch
from scipy.spatial.distance import cosine
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx


def compute_accuracy(pred, gt):
    n_matched = 0
    for i in range(pred.shape[0]):
        if pred[i].sum() > 0 and np.array_equal(pred[i], gt[i]):
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes


def compute_structural_score(pyg_graph):
    """
    Compute the average structural similarity score of
    the node embeddings of a graph.
    
    Args:
        pyg_graph (Data): A PyTorch Geometric Data object containing node embeddings and edge information.
    
    Returns:
        float: The average structural score of the graph.
    """
    # Extract node embeddings
    embeddings = pyg_graph.x.numpy()
    N = embeddings.shape[0]
    
    # Computing cosine similarities
    cosine_similarities = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                cosine_similarities[u, v] = 1 - cosine(embeddings[u], embeddings[v])

    # Compute degree differences
    degrees = degree(pyg_graph.edge_index[0]).numpy()
    degree_differences = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                degree_differences[u, v] = abs(degrees[u] - degrees[v])

    # Compute structural scores
    scores = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                scores[u, v] = cosine_similarities[u, v] / (1 + degree_differences[u, v])

    # Average the result
    total_score = np.sum(scores) / 2
    average_score = total_score / (N * (N - 1) / 2)

    return average_score


def compute_positional_score(pyg_graph):
    """
    Compute the average positional similarity score of
    the node embeddings of a graph.
    
    Args:
        pyg_graph (Data): A PyTorch Geometric Data object containing node embeddings and edge information.
    
    Returns:
        float: The average positional score of the graph.
    """
    # Extract node embeddings
    embeddings = pyg_graph.x.numpy()
    N = embeddings.shape[0]
    
    # Computing cosine similarities
    cosine_similarities = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                cosine_similarities[u, v] = 1 - cosine(embeddings[u], embeddings[v])
    
    # Convert to NetworkX to compute shortest paths
    G = to_networkx(pyg_graph)
    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Compute positional scores
    positional_scores = np.zeros((N, N))
    for u in range(N):
        for v in range(N):
            if u != v:
                path_len = shortest_path_lengths[u][v] if v in shortest_path_lengths[u] else float('inf')
                if path_len > 0 and path_len != float('inf'):
                    positional_scores[u, v] = cosine_similarities[u, v] / path_len
                else:
                    positional_scores[u, v] = 0.0
    
    # Get average positional score
    total_positional_score = np.sum(positional_scores) / 2
    average_positional_score = total_positional_score / (N * (N - 1) / 2)

    return average_positional_score


def compute_scores_for_graphs(data_list):
    """
    Compute structural and positional scores for a list of graphs.
    
    Args:
        data_list (list of Data):
            A list of PyTorch Geometric Data objects, each representing a graph.
    
    Returns:
        list(tuple): 
            A list of tuples where each tuple contains the structural 
            and the positional scores of a different graph.
            [(struct1, pos1), (struct2, pos2), ...]

    """
    graph_scores = []
    for data in data_list:
        struct_score = compute_structural_score(data)
        pos_score = compute_positional_score(data)
        graph_scores.append((struct_score, pos_score))
    return graph_scores


def compute_sim_prox_score(sim_mat, gt_mat, pred_mat):
    """
    Computes the average absolute distance between the similarity
    scores of the true alignments and the similarity scores of the 
    predicted alignments, considering only the rows and columns available
    in the groundtruth matrix (non-zero rows and columns).

    Args:
        sim_mat (np.ndarray):
            A 2D array of shape (N, M) where each cell (i, j) represents 
            a similarity score between the i-th row element and the j-th column element.
        gt_mat (np.ndarray): 
            A 2D binary array of shape (N, M) where 1 represents the true positive 
            alignments and 0 represents the true negatives. Can be incomplete.
        pred_mat (np.ndarray): 
            A 2D binary array of shape (N, M) where 1 represents the predicted positive 
            alignments and 0 represents the predicted negatives.

    Returns:
        float: The average similartity proximity score
    """
    abs_dists = []
    
    # Iterate over each row of gt_mat
    for i in range(gt_mat.shape[0]):
        # Find columns where there is a true alignment (1) in gt_mat
        true_indices = np.where(gt_mat[i] == 1)[0]
        
        for j in true_indices:
            # Compute absolute distances between sim_mat[i, j] and sim_mat[i, k] for k in pred_indices
            pred_indices = np.where(pred_mat[i] == 1)[0]
            abs_dists.extend(np.abs(sim_mat[i, j] - sim_mat[i, pred_indices]))

    # Compute the average absolute distance
    avg_dist = np.mean(abs_dists)
    
    return avg_dist


if __name__ == '__main__':
    # Test structural and positional scores:
    node_features_1 = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=torch.float)
    edge_index_1 = torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]], dtype=torch.long)
    data_1 = Data(x=node_features_1, edge_index=edge_index_1)

    node_features_2 = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float)
    edge_index_2 = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    data_2 = Data(x=node_features_2, edge_index=edge_index_2)

    data_list = [data_1, data_2]

    graph_scores = compute_scores_for_graphs(data_list)
    print("Graph Scores (struc, pos):", graph_scores)

    # Test similarity proximity score
    sim_mat = np.array([[0.1, 0.4, 0.3], [0.7, 0.5, 0.2], [0.6, 0.9, 0.8]])
    gt_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pred_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0]])

    sim_prox = compute_sim_prox_score(sim_mat, gt_mat, pred_mat)
    print("Similarity Proximity Score:", sim_prox)
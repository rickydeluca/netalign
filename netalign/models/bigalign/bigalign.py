import networkx as nx
import numpy as np
import numpy.matlib as matlib
import torch.nn as nn
from torch_geometric.utils import degree, to_networkx


class BigAlign(nn.Module):
    def __init__(self, cfg):
        """
        data1: object of Dataset class, contains information of source network
        data2: object of Dataset class, contains information of target network
        lamb: lambda
        """
        super(BigAlign, self).__init__()
        self.lamb = cfg.LAMBDA        

    @staticmethod
    def _build_degrees(graph):
        return degree(graph.edge_index[0], num_nodes=graph.num_nodes).tolist()

    @staticmethod
    def _build_clustering(graph):
        G = to_networkx(graph, to_undirected=True)
        cluster = nx.clustering(G)
        cluster_list = [cluster[i] for i in range(len(cluster))]    # convert to list
        return cluster_list

    def _extract_features(self, graph):
        """
        Preprocess input for unialign algorithms.
        """
        n_nodes = graph.num_nodes

        if graph.x is not None:
            nodes_degrees = self._build_degrees(graph)
            nodes_clustering = self._build_clustering(graph)

            if graph.x.shape[1] > 3:
                features = graph.x.dot(self.weight_features)
            else:
                features = graph.x

            N = np.zeros((n_nodes, 2+features.shape[1]))
            N[:,0 ] = nodes_degrees
            N[:,1 ] = nodes_clustering
            N[:,2:] = features

        else:
            N = np.zeros((n_nodes, 2))
            N[:,0] = self._build_degrees(graph)
            N[:,1] = self._build_clustering(graph)

        return N
 
    def align(self, pair_dict):
        self.source_graph = pair_dict['graph_pair'][0]
        self.target_graph = pair_dict['graph_pair'][1]
        

        if (self.source_graph.x is not None) and (self.target_graph.x is not None):
            self.weight_features = np.random.uniform(
                size=(self.source_graph.x.shape[1], 3)
            )

        N1 = self._extract_features(self.source_graph)
        N2 = self._extract_features(self.target_graph)
        lamb = self.lamb
        n2 = N2.shape[0]
        d = N2.shape[1]
        u, s, _ = np.linalg.svd(N1, full_matrices=False)

        # Transform S
        S = np.zeros((s.shape[0], s.shape[0]))

        for i in range(S.shape[1]):
            S[i, i] = s[i]
            S[i, i] = 1 / S[i, i] ** 2

        X = N1.T.dot(u).dot(S).dot(u.T)
        Y = lamb / 2 * np.sum(u.dot(S).dot(u.T), axis=0)
        P = N2.dot(X) - matlib.repmat(Y, n2, 1)

        return P.T, -1
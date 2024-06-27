import argparse
import os
import torch
import pdb

import numpy as np
import torch.nn as nn
from numpy import inf
from scipy.io import loadmat
from sklearn.preprocessing import normalize
from torch_geometric.utils import to_dense_adj, is_undirected


class FINAL(nn.Module):

    """
     Input:
       A1, A2: Input adjacency matrices with n1, n2 nodes
       N1, N2: Node attributes matrices, N1 is an n1*K matrix, N2 is an n2*K
             matrix, each row is a node, and each column represents an
             attribute. If the input node attributes are categorical, we can
             use one hot encoding to represent each node feature as a vector.
             And the input N1 and N2 are still n1*K and n2*K matrices.
             E.g., for node attributes as countries, including USA, China, Canada, 
             if a user is from China, then his node feature is (0, 1, 0).
             If N1 and N2 are emtpy, i.e., N1 = [], N2 = [], then no node
             attributes are input. 
    
       E1, E2: a L*1 cell, where E1{i} is the n1*n1 matrix and nonzero entry is
             the i-th attribute of edges. E2{i} is same. Similarly,  if the
             input edge attributes are categorical, we can use one hot
             encoding, i.e., E1{i}(a,b)=1 if edge (a,b) has categorical
             attribute i. If E1 and E2 are empty, i.e., E1 = {} or [], E2 = {}
             or [], then no edge attributes are input.
    
       H: a n2*n1 prior node similarity matrix, e.g., degree similarity. H
          should be normalized, e.g., sum(sum(H)) = 1.
       alpha: decay factor 
       maxiter, tol: maximum number of iterations and difference tolerance.
    
     Output: 
       S: an n2*n1 alignment matrix, entry (x,y) represents to what extend node-
        x in A2 is aligned to node-y in A1

    """

    def __init__(self, cfg):
        super(FINAL, self).__init__()
        self.alpha = cfg.ALPHA
        self.maxiter = cfg.MAXITER
        self.tol = cfg.TOL
        self.H_path = cfg.H

    def align(self, pair_dict):
        self.source_graph = pair_dict['graph_pair'][0]
        self.target_graph = pair_dict['graph_pair'][1]
        
        self.A1 = to_dense_adj(self.source_graph.edge_index).squeeze().cpu().numpy()
        self.A2 = to_dense_adj(self.target_graph.edge_index).squeeze().cpu().numpy()
        self.N1 = self.source_graph.x
        self.N2 = self.target_graph.x
        self.E1 = None # self._get_E(self.source_graph) 
        self.E2 = None # self._get_E(self.target_graph)
        self.H = self._get_H(self.H_path, self.source_graph, self.target_graph)

        self.alignment_matrix = None

        # If no node attributes input, then initialize as a vector of 1
        # so that all nodes are treated to have the same attributes which 
        # is equivalent to no given node attribute.
        # E1 should be (2, A1.shape[0], A1.shape[1])

        if self.N1 is None and self.N2 is None:
            self.N1 = np.ones((self.A1.shape[0], 1))
            self.N2 = np.ones((self.A2.shape[0], 1))

        if self.E1 is None and self.E2 is None:
            self.E1 = np.zeros((1, self.A1.shape[0], self.A1.shape[1]))
            self.E2 = np.zeros((1, self.A2.shape[0], self.A2.shape[1]))
            self.E1[0] = self.A1
            self.E2[0] = self.A2
        
        L = self.E1.shape[0]
        K = self.N1.shape[1]

        T1 = np.zeros_like(self.A1)
        T2 = np.zeros_like(self.A2)

        # Normalize edge feature vectors
        for i in range(L):
            T1 += self.E1[i] ** 2
            T2 += self.E2[i] ** 2

        for i in range(T1.shape[0]):
            for j in range(T1.shape[1]):
                if T1[i, j] > 0:
                    T1[i, j] = 1./T1[i, j]

        for i in range(T2.shape[0]):
            for j in range(T2.shape[1]):
                if T2[i, j] > 0:
                    T2[i, j] = 1./T2[i, j]

        for i in range(L):
            self.E1[i] = self.E1[i]*T1
            self.E2[i] = self.E2[i]*T2

        # Normalize node feature vectors
        self.N1 = normalize(self.N1)
        self.N2 = normalize(self.N2)

        # Compute node feature cosine cross-similarity
        n1 = self.A1.shape[0]
        n2 = self.A2.shape[0]
        N = np.zeros(n1 * n2)
        
        for k in range(K):
            N += np.kron(self.N1[:, k], self.N2[:, k])

        # Compute the Kronecker degree vector
        d = np.zeros_like(N)
        for i in range(L):
            for k in range(K):
                d += np.kron(np.dot(self.E1[i]*self.A1, self.N1[:, k]), np.dot(self.E2[i]*self.A2, self.N2[:, k]))

        D = N*d
        DD = 1./np.sqrt(D)
        DD[DD == inf] = 0

        # Fixed-point solution
        q = DD*N 
        h = self.H.flatten('F')
        s = h 

        for i in range(self.maxiter):
            print("iterations ", i + 1)
            prev = s
            M = (q*s).reshape((n2, n1), order='F' )
            S = np.zeros((n2, n1))
            for l in range(L):
                S += np.dot(np.dot(self.E2[l]*self.A2, M), self.E1[l]*self.A1)
            s = (1- self.alpha) * h + self.alpha * q * S.flatten('F')
            diff = np.sqrt(np.sum((s - prev)**2))
            print(diff)
            if diff < self.tol:
                break

        self.alignment_matrix = s.reshape((n1, n2))         
        return self.alignment_matrix, -1

    def get_alignment_matrix(self):
        if self.alignment_matrix is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.alignment_matrix
    
    @staticmethod
    def _get_H(path, source_graph, target_graph):
        """
        Get prior aligment matrix. If exist read it from file,
        otherwise generate a default one.
        """
        if path is None:    
            H = np.ones((source_graph.num_nodes, target_graph.num_nodes))
            H = H*(1/source_graph.num_nodes)
            return H
        else:    
            if not os.path.exists(path):
                raise Exception("Path '{}' is not exist".format(path))
            dict_H = loadmat(path)
            H = dict_H['H']
            return H

    @staticmethod
    def _get_E(graph):
        """
        Get the edge attribute as (1, N, N) matrix.
        """
        if graph.edge_attr is None:    
            return None
        else:    
            E = torch.zeros((graph.num_nodes, graph.num_nodes))
            undirected = is_undirected(graph.edge_index)

            # Populate the adjacency matrix with edge attributes
            for i in range(graph.edge_index.size(1)):
                source = graph.edge_index[0, i].item()
                target = graph.edge_index[1, i].item()
                weight = graph.edge_attr[i].item()
                
                E[source, target] = weight
                
                if undirected:
                    E[target, source] = weight

            # Reshapeto (1, N, N)
            return E.unsqueeze(0).detach().numpy()

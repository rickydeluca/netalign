import numpy as np
import torch.nn as nn
from numpy import inf
from torch_geometric.utils import to_dense_adj


class IsoRank(nn.Module):
    def __init__(self, H=None, alpha=0.82, maxiter=30, tol=1e-4):
        super(self, IsoRank).__init__()
        self.alpha = alpha
        self.maxiter = maxiter
        self.H = H
        self.tol = tol

    def forward(self, input_dict):
        # Read input dictionary.
        self.graph_s = input_dict['graph_s']
        self.graph_t = input_dict['graph_t']

        # Get and normalize adjacency matrices.
        A1 = to_dense_adj(self.graph_s.edge_index).cpu().numpy()
        A2 = to_dense_adj(self.graph_t.edge_index).cpu().numpy()

        n1 = A1.shape[0]
        n2 = A2.shape[0]

        d1 = 1 / A1.sum(axis=1)
        d2 = 1 / A2.sum(axis=1)

        d1[d1 == inf] = 0
        d2[d2 == inf] = 0
        d1 = d1.reshape(-1,1)
        d2 = d2.reshape(-1,1)

        W1 = d1*A1
        W2 = d2*A2
        
        # Map target to source.
        S = np.ones((n2,n1)) / (n1 * n2) 

        # Perform IsoRank.
        for iter in range(1, self.maxiter + 1):
            prev = S.flatten()
            if self.H is not None:
                S = (self.alpha*W2.T).dot(S).dot(W1) + (1-self.alpha) * self.H
            else:
                S = W2.T.dot(S).dot(W1)
            delta = np.linalg.norm(S.flatten()-prev, 2)
            print("Iteration: ", iter, " with delta = ", delta)
            if delta < self.tol:
                break
        
        # Return the predicted alignment matrix.
        self.S = S.T
        return self.S




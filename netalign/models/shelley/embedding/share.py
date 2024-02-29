import torch
import torch.nn as nn


class ShareEmbedding(nn.Module):
    def __init__(self, node_feature_dim):
        super(ShareEmbedding, self).__init__()
        self.node_feature_dim = node_feature_dim

    def forward(self, graph):
        return torch.ones((graph.num_nodes, self.node_feature_dim))
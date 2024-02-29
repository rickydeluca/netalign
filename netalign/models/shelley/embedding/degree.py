import torch.nn as nn
from torch_geometric.utils import degree


class DegreeEmbedding(nn.Module):
    def __init__(self, node_feature_dim):
        super(DegreeEmbedding, self).__init__()
        self.node_feature_dim = node_feature_dim

    def forward(self, graph):
            return degree(graph.edge_index[0],
                          num_nodes=graph.num_nodes).unsqueeze(1)
            
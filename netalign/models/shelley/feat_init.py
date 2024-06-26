import torch
import torch.nn as nn
from torch_geometric.utils import degree

def init_feat_module(cfg):
    # Feature initialization
    if cfg.TYPE.lower() == 'degree':
        f_init = Degree()
    elif cfg.TYPE.lower() == 'share':
        f_init = Share(cfg.FEATURE_DIM)
    else:
        raise ValueError(f"Invalid features: {cfg.TYPE}.")
    
    return f_init

class Degree(nn.Module):
    def __init__(self):
        super(Degree, self).__init__()

    def forward(self, graph):
        return degree(graph.edge_index[0], num_nodes=graph.num_nodes).unsqueeze(1)
    

class Share(nn.Module):
    def __init__(self, node_feature_dim):
        super(Share, self).__init__()
        self.node_feature_dim = node_feature_dim

    def forward(self, graph):
        return torch.ones((graph.num_nodes, self.node_feature_dim))
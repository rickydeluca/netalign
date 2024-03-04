from typing import Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.typing import SparseTensor


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 1, normalize: bool = True, bias: bool = True):
        super(GCN, self).__init__()

        self.conv_layers = nn.ModuleList()

        if num_layers == 1:
            self.conv_layers.append(GCNConv(in_channels, out_channels, normalize=normalize, bias=bias))
        else:
            self.conv_layers.append(GCNConv(in_channels, hidden_channels, normalize=normalize, bias=bias))
            for _ in range(1, num_layers - 1):
                self.conv_layers.append(GCNConv(hidden_channels, hidden_channels, normalize=normalize, bias=bias))
            self.conv_layers.append(GCNConv(hidden_channels, out_channels, normalize=normalize, bias=bias))

    def forward(self, x: Tensor, edge_index: Union[Tensor, SparseTensor],
                edge_weight: Optional[Tensor] = None) -> Tensor:
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index, edge_weight)
            x = F.relu(x)

        return x
    

class GCNMapping(nn.Module):
    def __init__(self, source_graph, target_graph, loss_fn, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 1, normalize: bool = True, bias: bool = True):
        super(GCNMapping, self).__init__()

        self.loss_fn = loss_fn
        self.source_graph = source_graph
        self.target_graph = target_graph
        self.gcn = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels,
                       num_layers=num_layers, normalize=normalize, bias=bias)
        
    def loss(self, source_batch, target_batch, gt_labels=None):
        hs = self.forward(x=self.source_graph.x,
                          edge_index=self.source_graph.edge_index,
                          edge_weight=self.source_graph.edge_attr)[source_batch]
        
        ht = self.forward(x=self.target_graph.x,
                          edge_index=self.target_graph.edge_index,
                          edge_weight=self.target_graph.edge_attr)[target_batch]
        
        batch_size = hs.shape[0]
        mapping_loss = self.loss_fn(hs, ht, source_batch, target_batch, gt_labels) / batch_size

        return mapping_loss
    
    
    def forward(self, x: Tensor, edge_index: Union[Tensor, SparseTensor],
                edge_weight: Optional[Tensor] = None) -> Tensor:
        return self.gcn(x, edge_index, edge_weight)
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
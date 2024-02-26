from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential, Tanh
from torch_geometric.nn import GCNConv, GINEConv
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
    

class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, bias=True):
        super(GIN, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.bn_in = torch.nn.BatchNorm1d(in_channels)

        self.nn1 = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv1 = GINEConv(self.nn1, eps=eps, train_eps=train_eps, edge_dim=in_channels)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.nn2 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv2 = GINEConv(self.nn2, eps=eps, train_eps=train_eps, edge_dim=in_channels)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.nn3 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv3 = GINEConv(self.nn3, eps=eps, train_eps=train_eps, edge_dim=in_channels)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.nn4 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv4 = GINEConv(self.nn4, eps=eps, train_eps=train_eps, edge_dim=in_channels)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.nn5 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv5 = GINEConv(self.nn5, eps=eps, train_eps=train_eps, edge_dim=in_channels)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, out_channels, bias=False)

    def forward(self, X, edge_index, edge_attr):
        x = self.bn_in(X)
        xs = [x]

        xs.append(self.conv1(xs[-1], edge_index, edge_attr))
        xs.append(self.conv2(xs[-1], edge_index, edge_attr))
        # xs.append(self.conv3(xs[-1], edge_index, edge_attr))
        # xs.append(self.conv4(xs[-1], edge_index, edge_attr))
        # xs.append(self.conv5(xs[-1], edge_index, edge_attr))
        xs.append(torch.tanh(self.fc1(xs[-1])))

        x = torch.cat(xs[1:], dim=-1)

        return x

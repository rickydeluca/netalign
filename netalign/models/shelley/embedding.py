from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential, Tanh
from torch_geometric.nn import GCNConv, GINEConv
from torch_geometric.typing import SparseTensor


class GINE(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_conv_layers=1, bias=True):
        super(GINE, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.bn_in = nn.BatchNorm1d(in_channels)

        self.num_conv_layers = num_conv_layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # First conv layer
        self.nn = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv_layers.append(GINEConv(self.nn, eps=eps, train_eps=train_eps, edge_dim=in_channels))
        self.bn_layers.append(nn.BatchNorm1d(dim))

        # Remaning conv layers
        for _ in range(1, num_conv_layers):
            self.nn = Sequential(Linear(dim, dim, bias=bias), act)
            self.conv_layers.append(GINEConv(self.nn, eps=eps, train_eps=train_eps, edge_dim=in_channels))
            self.bn_layers.append(nn.BatchNorm1d(dim))

        # Final linar layer
        self.fc = Linear(dim, out_channels, bias=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.bn_in(x)
        xs = [x]

        # Convolution
        for i in range(self.num_conv_layers):
            xs.append(self.bn_layers[i](self.conv_layers[i](xs[-1], edge_index, edge_attr)))

        # Linear
        xs.append(torch.tanh(self.fc(xs[-1])))

        x = torch.cat(xs[1:], dim=-1)
        
        return x
    

class GINE2(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_conv_layers=1, bias=True):
        super(GINE2, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.bn_in = nn.BatchNorm1d(in_channels)

        self.num_conv_layers = num_conv_layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # First conv layer
        self.nn = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv_layers.append(GINEConv(self.nn, eps=eps, train_eps=train_eps, edge_dim=in_channels))
        self.bn_layers.append(nn.BatchNorm1d(dim))

        # Remaning conv layers
        for _ in range(1, num_conv_layers):
            self.nn = Sequential(Linear(dim, dim, bias=bias), act)
            self.conv_layers.append(GINEConv(self.nn, eps=eps, train_eps=train_eps, edge_dim=in_channels))
            self.bn_layers.append(nn.BatchNorm1d(dim))

        # Final linar layer
        self.fc = Linear(dim, out_channels, bias=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.bn_in(x)

        # Convolution
        for i in range(self.num_conv_layers):
            x = self.bn_layers[i](self.conv_layers[i](x, edge_index, edge_attr))

        # Linear
        x = torch.tanh(self.fc(x))
        
        return x
    

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, normalize=True, bias=True):
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
                edge_attr: Optional[Tensor] = None) -> Tensor:
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index, edge_attr)
            x = F.relu(x)

        return x
    


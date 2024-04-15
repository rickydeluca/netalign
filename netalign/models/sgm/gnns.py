from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, Sequential, Tanh
from torch_geometric.nn import GCNConv, GINConv, GINEConv
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
                edge_attr: Optional[Tensor] = None) -> Tensor:
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x, edge_index, edge_attr)
            x = F.relu(x)

        return x
    

class GINE(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_conv_layers=2, bias=True):
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


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, bias=True):
        super(GIN, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.bn_in = torch.nn.BatchNorm1d(in_channels)

        self.nn1 = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv1 = GINConv(self.nn1, eps=eps, train_eps=train_eps)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        self.nn2 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv2 = GINConv(self.nn2, eps=eps, train_eps=train_eps)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        self.nn3 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv3 = GINConv(self.nn3, eps=eps, train_eps=train_eps)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.nn4 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv4 = GINConv(self.nn4, eps=eps, train_eps=train_eps)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        self.nn5 = Sequential(Linear(dim, dim, bias=bias), act)
        self.conv5 = GINConv(self.nn5, eps=eps, train_eps=train_eps)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, out_channels, bias=False)

    def forward(self, X, X_importance, edge_index):
        x = self.bn_in(X * X_importance)
        xs = [x]

        xs.append(self.conv1(xs[-1], edge_index))
        xs.append(self.conv2(xs[-1], edge_index))
        xs.append(self.conv3(xs[-1], edge_index))
        xs.append(self.conv4(xs[-1], edge_index))
        xs.append(self.conv5(xs[-1], edge_index))
        xs.append(torch.tanh(self.fc1(xs[-1])))

        x = torch.cat(xs[1:], dim=-1)

        return x
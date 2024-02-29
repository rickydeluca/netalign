import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, Tanh
from torch_geometric.nn import GINEConv


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

    def forward(self, X, edge_index, edge_attr):
        x = self.bn_in(X)
        xs = [x]

        # Convolution
        for i in range(self.num_conv_layers):
            xs.append(self.bn_layers[i](self.conv_layers[i](xs[-1], edge_index, edge_attr)))

        # Linear
        xs.append(torch.tanh(self.fc(xs[-1])))

        # x = torch.cat(xs[1:], dim=-1)
        x = xs[-1]
        
        return x
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, Tanh
from torch_geometric.nn import GINConv, GINEConv


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

class GIN2(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_conv_layers=1, bias=True):
        super(GIN2, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.bn_in = nn.BatchNorm1d(in_channels)

        self.num_conv_layers = num_conv_layers
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        # First conv layer
        self.nn = Sequential(Linear(in_channels, dim, bias=bias), act)
        self.conv_layers.append(GINConv(self.nn, eps=eps, train_eps=train_eps))
        self.bn_layers.append(nn.BatchNorm1d(dim))

        # Remaning conv layers
        for _ in range(1, num_conv_layers):
            self.nn = Sequential(Linear(dim, dim, bias=bias), act)
            self.conv_layers.append(GINConv(self.nn, eps=eps, train_eps=train_eps))
            self.bn_layers.append(nn.BatchNorm1d(dim))

        # Final linar layer
        self.fc = Linear(dim, out_channels, bias=False)

    def forward(self, x, x_importance, edge_index):
        x = self.bn_in(x * x_importance)

        # Convolution
        for i in range(self.num_conv_layers):
            x = self.bn_layers[i](self.conv_layers[i](x, edge_index))

        # Linear
        x = torch.tanh(self.fc(x))
        
        return x
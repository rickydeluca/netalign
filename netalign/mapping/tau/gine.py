import torch
from torch.nn import Sequential, Linear, Tanh
from torch_geometric.nn import GINEConv


class GINE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_layers=2, bias=True):
        super(GINE, self).__init__()

        act = Tanh()
        eps = 0.0
        train_eps = False

        self.bn_in = torch.nn.BatchNorm1d(in_channels)

        self.num_layers = num_layers
        self.conv = []
        self.bn = []

        for i in range(num_layers):
            if i == 0:
                nn = Sequential(Linear(in_channels, dim, bias=bias), act)
            else:
                nn = Sequential(Linear(dim, dim, bias=bias), act)

            self.conv.append(GINEConv(nn, eps=eps, train_eps=train_eps, edge_dim=1))
            self.bn.append(torch.nn.BatchNorm1d(dim))

        self.fc1 = Linear(dim, out_channels, bias=False)

    def forward(self, X, edge_index, edge_attr=None):
        x = self.bn_in(X)
        xs = [x]

        # Convolve
        for i in range(self.num_layers):
            xs.append(self.bn[i](self.conv[i](xs[-1], edge_index, edge_attr=edge_attr)))

        # Linear layer
        xs.append(torch.tanh(self.fc1(xs[-1])))

        x = torch.cat(xs[1:], dim=-1)

        return x

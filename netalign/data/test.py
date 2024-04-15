import torch
from torch_geometric.loader import DataLoader

from netalign.data.dataset import BrainNetworkRobust, BrainNetworkTopology

dataset = BrainNetworkRobust(root_dir='data/bn1000',
                              ages=['6m'],
                              p_add=0.1,
                              p_rm=0.0,
                              num_copies=2,
                              train_ratio=0.20,
                              val_ratio=0.0)
print(len(dataset))

dataloader = DataLoader(dataset, batch_size=1)

for i, pair in enumerate(dataloader):
    print(f'\n\npair {i}')
    print(pair)

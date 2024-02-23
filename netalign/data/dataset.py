from typing import Optional

import networkx as nx
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

import netalign.data.utils as utils


class SemiSyntheticDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 p_add: float = 0.0,
                 p_rm: float = 0.0,
                 size: int = 100,
                 train_ratio: float = 0.2,
                 seed: Optional[int] = None):
        
        super(SemiSyntheticDataset, self).__init__()
  
        self.p_add = p_add              # Prob. of adding an edge
        self.p_rm = p_rm                # Prob. of removing an edge
        self.train_ratio = train_ratio  # Percentage of nodes to use for training
        self.seed = seed                # Random seed
        self.size = size                # Number of target copies

        # Load PyG graph with initialized node features
        self.source_pyg = utils.edgelist_to_pyg(root_dir, seed=seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.get_pair(idx)

    def get_pair(self, idx):
        """
        Generate a random semi-synthetic copy of the source input graph
        and return a dictionary with the source-target graph pair in
        pytorch geometric data format and all the correspoinding informations.
        """
        
        # Generate random semi-synthetic pyg target graph
        target_pyg, node_mapping = utils.generate_target_graph(
            self.source_pyg,
            p_add=self.p_add,
            p_rm=self.p_rm
        )

        # Split alignments in train and test subsets
        train_dict, test_dict = utils.shuffle_and_split(node_mapping,
                                                        split_ratio=self.train_ratio)

        # Assemble pair informations in a dictionary
        pair_dict = edict()
        pair_dict = {
            'graph_pair': [self.source_pyg, target_pyg],
            'train_dict': train_dict,                       # Dictionary with groundtruth training alignments
            'test_dict': test_dict                          # Dictionary with groundtruth test alignments
        }
        
        return pair_dict
    

if __name__ == '__main__':
    # Test Dataset
    data_dir = 'data/ppi'
    gm_dataset = SemiSyntheticDataset(root_dir=data_dir,
                                      p_add=0.5,
                                      p_rm=0.0,
                                      size=100,
                                      train_ratio=0.2)
    
    print('-'*20)
    print("Dataset size:", len(gm_dataset))

    first_elem = next(iter(gm_dataset))
    print("First element of the dataset: ", first_elem)
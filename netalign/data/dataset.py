import json
import os
from typing import Optional

import networkx as nx
import numpy as np
from easydict import EasyDict as edict
from networkx.readwrite import json_graph
from scipy.io import loadmat
from torch.utils.data import Dataset as TorchDataset

import netalign.data.utils as utils


class BaseDataset:
    """
    This class receives input from edgelist, creates from it the GraphSAGE format 
    with predefined folder structure.
    The data folder must contains these files:
        - G.json
        - id2idx.json
        - features.npy (optional)

    Args:
        data_dir: Data directory which contains files mentioned above.
    """

    def __init__(self, data_dir, verbose=True):
        # Convert edgelist to graphsage format
        self.data_dir = data_dir
        self._edgelist_to_graphsage()

        # Read graphsage
        self.graphsage_dir = data_dir + "/graphsage"
        self._load_G()
        self._load_id2idx()
        self._load_features()
        self.load_edge_features()
        
        if verbose:
            print("Dataset info:")
            print("- Nodes: ", len(self.G.nodes()))
            print("- Edges: ", len(self.G.edges()))

    def _edgelist_to_graphsage(self):
        utils.edgelist_to_graphsage(self.data_dir)

    def _load_G(self):
        G_data = json.load(open(os.path.join(self.graphsage_dir, "G.json")))
        self.G = json_graph.node_link_graph(G_data)
        if type(list(self.G.nodes())[0]) is int:
            mapping = {k: str(k) for k in self.G.nodes()}
            self.G = nx.relabel_nodes(self.G, mapping)

    def _load_id2idx(self):
        id2idx_file = os.path.join(self.graphsage_dir, 'id2idx.json')
        conversion = type(list(self.G.nodes())[0])
        self.id2idx = {}
        id2idx = json.load(open(id2idx_file))
        for k, v in id2idx.items():
            self.id2idx[conversion(k)] = v

    def _load_features(self):
        self.features = None
        feats_path = os.path.join(self.graphsage_dir, 'feats.npy')
        if os.path.isfile(feats_path):
            self.features = np.load(feats_path)
        else:
            self.features = None
        return self.features

    def load_edge_features(self):
        self.edge_features= None
        feats_path = os.path.join(self.graphsage_dir, 'edge_feats.mat')
        if os.path.isfile(feats_path):
            edge_feats = loadmat(feats_path)['edge_feats']
            self.edge_features = np.zeros((len(edge_feats[0]),
                                           len(self.G.nodes()),
                                           len(self.G.nodes())))
            for idx, matrix in enumerate(edge_feats[0]):
                self.edge_features[idx] = matrix.toarray()
        else:
            self.edge_features = None
        return self.edge_features

    def get_adjacency_matrix(self):
        return utils.construct_adjacency(self.G, self.id2idx)

    def get_nodes_degrees(self):
        return utils.build_degrees(self.G, self.id2idx)

    def get_nodes_clustering(self):
        return utils.build_clustering(self.G, self.id2idx)

    def get_edges(self):
        return utils.get_edges(self.G, self.id2idx)

    def check_id2idx(self):
        for i, node in enumerate(self.G.nodes()):
            if (self.id2idx[node] != i):
                print("Failed at node %s" % str(node))
                return False
        return True


class SemiSyntheticDataset(TorchDataset):
    def __init__(self,
                 root_dir: str,
                 p_add: float = 0.0,
                 p_rm: float = 0.0,
                 size: int = 100,
                 train_ratio: float = 0.2,
                 seed: Optional[int] = None):
        
        super(SemiSyntheticDataset, self).__init__()

        self.source_dataset = BaseDataset(root_dir, verbose=False)   
        self.p_add = p_add              # Prob. of adding an edge
        self.p_rm = p_rm                # Prob. of removing an edge
        self.train_ratio = train_ratio  # Percentage of nodes to use for training
        self.seed = seed                # Random seed
        self.size = size                # Number of target copies
            

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
        
        # Generate the source pyg graph
        source_pyg = utils.to_pyg_graph(
            G=self.source_dataset.G,
            id2idx=self.source_dataset.id2idx,
            node_feats=self.source_dataset.features,
            edge_feats=self.source_dataset.edge_features,
            node_metrics=[],
            edge_metrics=[]
        )
        
        # Generate random semi-synthetic pyg target graph
        target_pyg, groundtruth = utils.generate_synth_clone(
            source_pyg,
            p_add=self.p_add,
            p_rm=self.p_rm
        )
        
        # Split alignments in train and test subsets
        train_dict, test_dict = utils.train_test_split(groundtruth,
                                                       split_ratio=self.train_ratio)
            

        # Assemble pair informations in a dictionary
        pair_dict = edict()
        pair_dict = {
            'graph_pair': [source_pyg, target_pyg],
            'train_dict': train_dict,   # Dictionary with groundtruth training alignments
            'test_dict': test_dict      # Dictionary with groundtruth test alignments
        }
        
        return pair_dict
    

# --- Test dataset ---     
if __name__ == '__main__':
    data_dir = 'data/edi3'
    gm_dataset = SemiSyntheticDataset(root_dir=data_dir,
                                      p_add=0.2,
                                      p_rm=0.2,
                                      size=100,
                                      train_ratio=0.2)
    
    print('-'*20)
    print("Dataset size:", len(gm_dataset))

    first_elem = next(iter(gm_dataset))
    print("First element of the dataset: ", first_elem)
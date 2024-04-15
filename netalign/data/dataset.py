import glob
import os
from typing import List, Optional, Union

import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset

import netalign.data.utils as utils


class RobustnessDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 subset: Optional[List[str]] = None,
                 train_ratio: float = 0.2,
                 val_ratio: float = 0.0,
                 p_add: float = 0.0,
                 p_rm: float = 0.0,
                 num_copies: int = 5,
                 seed: Optional[int] = None,
                 gt_mode: str = 'matrix',
                 permute: bool = True):
        
        super(RobustnessDataset, self).__init__()

        self.root_dir = root_dir
        self.subset = subset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.p_add = p_add
        self.p_rm = p_rm
        self.num_copies = num_copies
        self.gt_mode = gt_mode
        self.permute = permute

        # Build alignment pairs and the corrisponding groundtruths.
        self.alignment_pairs = []
        self.groundtruths = []

        if self.gt_mode == 'matrix':
            self.train_gts = []
            self.val_gts = []
            self.test_gts = []
        else:
            self.train_dicts = []
            self.val_dicts = []
            self.test_dicts = []

        if subset is None:
            self.subset = ['*'] # Use all edgelist within the root dir.
            
        for _ in range(num_copies):
            for net_name in self.subset:
                matching_files = glob.glob(os.path.join(root_dir, f"*{net_name}*"))
                for edgelist in matching_files:
                    graph_s, _ = utils.edgelist_to_pyg(edgelist)

                    graph_t, node_mapping = utils.generate_target_graph(
                        pyg_source=graph_s,
                        p_rm=self.p_rm,
                        p_add=self.p_add,
                        mapping_type=self.gt_mode,
                        permute=self.permute
                    )

                    self.alignment_pairs.append((graph_s, graph_t))
                    self.groundtruths.append(node_mapping)

                    if self.gt_mode == 'matrix':
                        train_gt, val_gt, test_gt = utils.split_groundtruth_matrix(
                            node_mapping,
                            train_ratio=self.train_ratio,
                            val_ratio=self.val_ratio
                        )

                        self.train_gts.append(train_gt)
                        self.val_gts.append(val_gt)
                        self.test_gts.append(test_gt)
                    else:
                        # Split alignments in train, val and test subsets.
                        train_dict, val_dict, test_dict = utils.shuffle_and_split_dict(
                            node_mapping,
                            train_ratio=self.train_ratio,
                            val_ratio=self.val_ratio)
                        
                        self.train_dicts.append(train_dict)
                        self.val_dicts.append(val_dict)
                        self.test_dicts.append(test_dict)

    
    def __len__(self):
        return len(self.alignment_pairs)
    
    def __getitem__(self, idx):
        # Get pair attributes
        graph_s, graph_t = self.alignment_pairs[idx]

        if self.gt_mode == 'matrix':
            gt_perm = self.groundtruths[idx]
            train_gt = self.train_gts[idx]
            val_gt = self.val_gts[idx]
            test_gt = self.test_gts[idx]

            # Build pair dict
            pair_dict = edict()
            pair_dict = {
                'graph_pair': [graph_s, graph_t],
                'num_nodes': [torch.tensor(graph_s.num_nodes), torch.tensor(graph_t.num_nodes)],
                'gt_perm': gt_perm,
                'train_gt': train_gt,
                'val_gt': val_gt,
                'test_gt': test_gt
            }

        else:
            train_dict = self.train_dicts[idx]
            test_dict = self.test_dicts[idx]
            val_dict = self.val_dicts[idx]

            pair_dict = {
                'graph_pair': [graph_s, graph_t],
                'train_dict': train_dict,
                'val_dict': val_dict,
                'test_dict': test_dict
            }

        return pair_dict



class TopologyDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 source_names: Optional[List[str]] = None,
                 target_names: Optional[List[str]] = None,
                 train_ratio: float = 0.0,
                 val_ratio: float = 0.0,
                 num_copies: int = 30,
                 seed: Optional[int] = None):
        
        super(TopologyDataset, self).__init__()

        self.source_names = source_names
        self.target_names = target_names
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        # Build alignmnet pairs
        source_net_lists = []   
        target_net_lists = []
        for net_name in self.source_names:
            source_net_lists.append(glob.glob(os.path.join(root_dir, f"*{net_name}*")))

        for net_name in self.target_names:
            target_net_lists.append(glob.glob(os.path.join(root_dir, f"*{net_name}*")))

        if len(source_net_lists) != len(target_net_lists):
            raise ValueError('The list of source networks must be equal to the list of target networks.')
        
        self.alignment_pairs = []
        self.groundtruths = []
        self.train_masks = []
        self.val_masks = []
        self.test_masks = []

        for _ in num_copies:
            for i in enumerate(source_net_lists):
                for edgelist1 in source_net_lists[i]:
                    for edgelist2 in target_net_lists[i]:
                        # Read graphs
                        graph_s, id2idx_s = utils.edgelist_to_pyg(edgelist1)
                        graph_t, id2idx_t = utils.edgelist_to_pyg(edgelist2)

                        # Get groundtruths
                        gt_matrix = utils.create_alignment_matrix(id2idx_s, id2idx_t)

                        train_mask, val_mask, test_mask = utils.generate_split_masks(
                            gt_matrix,
                            train_ratio=self.train_ratio,
                            val_ratio=self.val_ratio
                        )

                        self.alignment_pairs.append((graph_s, graph_t))
                        self.groundtruths.append(gt_matrix)
                        self.train_masks.append(train_mask)
                        self.val_masks.append(val_mask)
                        self.test_masks.append(test_mask)
        
    def __len__(self):
        return len(self.alignment_pairs)
    
    def __getitem__(self, idx):
        # Get pair attributes
        graph_s, graph_t = self.alignment_pairs[idx]
        gt_matrix = self.groundtruths[idx]
        train_mask = self.train_masks[idx]
        val_mask = self.val_masks[idx]
        test_mask = self.test_masks[idx]

        # Build pair dict
        pair_dict = edict()
        pair_dict = {
            'graph_pair': [graph_s, graph_t],
            'gt_matrix': gt_matrix,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }

        return pair_dict


class RealDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 train_ratio: float = 0.2,
                 size: int = 10,
                 seed: Optional[int] = None):
        
        super(RealDataset, self).__init__()

        self.size = size
        self.train_ratio = train_ratio  # Percentage of nodes to use for training
        self.seed = seed                # Random seed

        # Load PyG graph with initialized node features
        self.source_pyg, self.source_id2idx = utils.edgelist_to_pyg(f'{root_dir}/src', seed=seed)
        self.target_pyg, self.target_id2idx = utils.edgelist_to_pyg(f'{root_dir}/tgt', seed=seed)

        # Load groundtruth dict
        self.groundtruth_dict = utils.read_dict(f'{root_dir}/groundtruth', self.source_id2idx, self.target_id2idx)

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

        # Split groundtruth in train and test subsets
        train_dict, test_dict = utils.shuffle_and_split(self.groundtruth_dict,
                                                        split_ratio=self.train_ratio)

        # Assemble pair informations in a dictionary
        pair_dict = edict()
        pair_dict = {
            'graph_pair': [self.source_pyg, self.target_pyg],
            'train_dict': train_dict,
            'test_dict': test_dict
        }
        
        return pair_dict


class SemiSyntheticDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 p_add: float = 0.0,
                 p_rm: float = 0.0,
                 size: int = 100,
                 train_ratio: float = 0.2,
                 val_ratio: Optional[float] = None,
                 seed: Optional[int] = None):
        
        super(SemiSyntheticDataset, self).__init__()
  
        self.p_add = p_add              # Prob. of adding an edge
        self.p_rm = p_rm                # Prob. of removing an edge
        self.train_ratio = train_ratio  # Percentage of nodes to use for training
        self.val_ratio = val_ratio      # Percentage of nodes to use for validation
        self.seed = seed                # Random seed
        self.size = size                # Number of target copies

        # Load PyG graph with initialized node features
        self.source_pyg, _ = utils.edgelist_to_pyg(root_dir, seed=seed)

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
        train_dict, val_dict, test_dict = utils.shuffle_and_split_dict(node_mapping,
                                                                       train_ratio=self.train_ratio,
                                                                       val_ratio=self.val_ratio)

        # Assemble pair informations in a dictionary
        pair_dict = edict()
        pair_dict = {
            'graph_pair': [self.source_pyg, target_pyg],
            'train_dict': train_dict,                       # Groundtruth training alignments
            'val_dict': val_dict,                           # Groundtruth validation alignments
            'test_dict': test_dict                          # Groundtruth test alignments
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
import argparse
import random

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from netalign.data.dataset import RealDataset
from netalign.data.utils import dict_to_perm_mat, move_tensors_to_device
from netalign.evaluation.matchers import greedy_match
from netalign.evaluation.metrics import compute_accuracy
from netalign.models import init_align_model


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description='Read command line arguments.')

    parser.add_argument('-s', '--source', type=str, required=True, help="Path to source network edgelist.")
    parser.add_argument('-t', '--target', type=str, required=True, help="Path to target network edgelist.")
    parser.add_argument('-c', '--cfg', type=str, required=True, help="Path to the model configuration (YAML file).")
    parser.add_argument('--num_exps', type=int, default=10, help="Number of experiments. Default: 10")
    parser.add_argument('--train_ratio', type=float, default=0.2, help="Percentage of nodes/graphs to use as training set. Default: 0.2")
    parser.add_argument('--val_ratio', type=float, default=0.0, help="Percentage of nodes/graphs to use as validation set. Default: 0.0")
    parser.add_argument('--seed', type=int, default=None, help="Seed for reproducibility. Default: None")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use. Default: cuda")
    parser.add_argument('--out_dir', type=str, default='results2/degree_omology', help="Path to the directory where to save the test results. Default: results2/degree_omology")
    
    return parser.parse_args()

def read_config_file(yaml_file):
    """
    Read the yaml configuration file and return it
    as dictionary.
    """
    with open(yaml_file) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    return cfg


def get_omology_degree_seeds(pyg_graphs, id2idx, idx2id, p=0.2):
    """
    Generate a mapping between the nodes of source
    and target networks using the name omology of the nodes 
    but keep only the pairs with an high degree value.

    Select only the `p` percentage of this mapping as 
    seed nodes for trainable algorithms.
    """
    source_pyg = pyg_graphs[0]
    target_pyg = pyg_graphs[1]
    source_id2idx = id2idx[0]
    target_id2idx = id2idx[1]
    source_idx2id = idx2id[0]
    target_idx2id = idx2id[1]
    
    # Get num of seeds
    num_seeds = int(source_pyg.num_nodes * p)

    # Sort by degree
    source_degrees = degree(source_pyg.edge_index[0])
    target_degrees = degree(target_pyg.edge_index[0])

    source_indices = source_degrees.argsort(descending=True)
    target_indices = target_degrees.argsort(descending=True)

    # Keep top-p nodes
    source_indices = source_indices[:num_seeds]
    target_indices = target_indices[:num_seeds]

    # Get alignment by omology
    seed_dict = {}
    for s_idx in source_indices:
        s_id = source_idx2id[s_idx.item()][0]
        
        # Check if both source and target have the node
        if s_id in list(target_id2idx.keys()):
            t_idx = target_id2idx[s_id]

            # Check if the target node is in top degree tier
            if t_idx in target_indices:
                seed_dict[s_idx.item()] = t_idx.item()
        
    return seed_dict


def get_test_aligns(id2idx, seed_aligns):
    """
    Get test alignments dictionary using node names omology
    but excludes from the list the nodes already used as seeds.
    """

    source_id2idx = id2idx[0]
    target_id2idx = id2idx[1]

    test_aligns = {}

    for s_id, s_idx in source_id2idx.items():
        if s_id in target_id2idx.keys() and s_idx.item() not in seed_aligns.keys():
            test_aligns[s_idx.item()] = target_id2idx[s_id].item()

    return test_aligns


if __name__ == '__main__':
    # Read input
    args = parse_args()
    cfg = read_config_file(args.cfg)

    # Set reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Load dataset
    dataset = RealDataset(
        source_path=args.source,
        target_path=args.target,
        gt_path=None,
        gt_mode='omology',
        seed=args.seed
    )

    dataloader = DataLoader(dataset, shuffle=False)

    # Init model
    model, model_name = init_align_model(cfg)

    # Run experiments
    pair_dict = next(iter(dataloader))
    accs = []
    best_acc = 0
    best_align_matrix = None
    for exp_id in range(1, args.num_exps+1):
        # Generate set of seed nodes such as node pairs
        # with same name and high degree
        seed_aligns = get_omology_degree_seeds(pair_dict['graph_pair'],
                                               pair_dict['id2idx'],
                                               pair_dict['idx2id'],
                                               p=0.2)
        
        # Generate test groundtruth by omology but 
        # exlude the train alignments
        test_aligns = get_test_aligns(pair_dict['id2idx'], seed_aligns)

        pair_dict['gt_train'] = seed_aligns
        pair_dict['gt_test'] = test_aligns

        # Move to device
        pair_dict = move_tensors_to_device(pair_dict, device)
        try:
            model = model.to(device)
        except: 
            pass

        # Align
        S, _ = model.align(pair_dict)
        P = greedy_match(S)

        # Compute accuracy
        gt_test = dict_to_perm_mat(pair_dict['gt_test'], pair_dict['graph_pair'][0].num_nodes, pair_dict['graph_pair'][1].num_nodes).detach().cpu().numpy()
        acc = compute_accuracy(P, gt_test)

        print(f"\nExperiment: {exp_id}, Accuracy: {acc}")

        # Save best prediction
        if acc > best_acc:
            best_acc = acc
            best_align_matrix = P

        accs.append(acc)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print(f"\n\nMean accuracy: {mean_acc}, Standard deviation: {std_acc}")

    # Save best predicted aligns
    source_net_name = args.source.split('/')[-1].split('.')[0]
    target_net_name = args.target.split('/')[-1].split('.')[0]

    nonzero_rows, nonzero_cols = best_align_matrix.nonzero()

    with open(f'output/avg_{source_net_name}-{target_net_name}.txt', 'w') as file:
        for src_idx, tgt_idx in zip(nonzero_rows, nonzero_cols):
            src_id = pair_dict['idx2id'][0][src_idx]
            tgt_id = pair_dict['idx2id'][1][tgt_idx]
            file.write(f"{src_id} {tgt_id}\n")
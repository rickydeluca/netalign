import argparse
import csv
import os
import time
import random

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader

from netalign.data.dataset import SemiSyntheticDataset
from netalign.data.utils import move_tensors_to_device, dict_to_perm_mat
from netalign.evaluation.matchers import greedy_match
from netalign.evaluation.metrics import compute_accuracy, compute_sim_prox_score
from netalign.models import init_align_model


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description='Read the test configuration.')

    parser.add_argument('-c', '--cfg', type=str, required=True, help="Path to the model configuration (YAML file).")
    parser.add_argument('-s', '--source_path', type=str, required=True, help="Path to the source network in edgelist format.")
    parser.add_argument('--size', type=int, default=100, help="Number of random target network copies. Default: 100")
    parser.add_argument('--train_ratio', type=float, default=0.2, help="Percentage of nodes/graphs to use as training set. Default: 0.2")
    parser.add_argument('--val_ratio', type=float, default=0.0, help="Percentage of nodes/graphs to use as validation set. Default: 0.0")
    parser.add_argument('--seed', type=int, default=None, help="Seed for reproducibility. Default: None")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for training. Default: cuda")
    parser.add_argument('--res_dir', type=str, default='results/robustness', help="Path to the directory where to save the test results. Default: results/robustness")
    
    return parser.parse_args()


def read_config_file(yaml_file):
    """
    Read the yaml configuration file and return it
    as an `EasyDict` dictionary.
    """
    with open(yaml_file) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    return cfg

if __name__ == '__main__':
    # Read configuration file
    args = parse_args()
    cfg = read_config_file(args.cfg)

    # Set reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Init result file
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    
    data_name = args.source_path.split('/')[-1].split('.')[0]
    _, model_name = init_align_model(cfg)

    res_file = f'{args.res_dir}/{model_name}_{data_name}.csv'

    header = ['model', 'data', 'size', 'p_add', 'p_rm', 'avg_acc', 'std_acc', 'avg_sim_prox', 'std_sim_prox', 'avg_time', 'avg_best_epoch']
    with open(res_file, 'w') as rf:
        csv_writer = csv.DictWriter(rf, fieldnames=header)
        csv_writer.writeheader()
    
    # Run tests
    noise_types = ['add', 'rm']
    noise_probs = [0.00, 0.05, 0.10, 0.20, 0.25]

    for noise_type in noise_types:
        for noise_prob in noise_probs:
            p_add = noise_prob if noise_type == 'add' else 0.0
            p_rm = noise_prob if noise_type == 'rm' else 0.0

            dataset = SemiSyntheticDataset(
                source_path=args.source_path,
                size=args.size,
                permute=True,
                p_add=p_add,
                p_rm=p_rm,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed
            )
            
            dataloader = DataLoader(dataset, shuffle=True)

            matching_accs = []
            sim_proxs = []
            comp_times = []
            best_epochs = []

            for pair_id, pair_dict in enumerate(dataloader):
                # Init model
                model, _ = init_align_model(cfg)

                # Move to device
                pair_dict = move_tensors_to_device(pair_dict, device)
                try:
                    model = model.to(device)
                except:
                    pass

                # Predict alignment
                start_time = time.time()
                S, best_epoch = model.align(pair_dict)
                elapsed_time = time.time() - start_time

                # Compute matching accuracy
                P = greedy_match(S)
                gt_test = dict_to_perm_mat(pair_dict['gt_test'], pair_dict['graph_pair'][0].num_nodes, pair_dict['graph_pair'][1].num_nodes).detach().cpu().numpy()
                acc = compute_accuracy(P, gt_test)

                # Compute similarity proximity score
                sim_prox = compute_sim_prox_score(S, gt_test, P)

                print(f"\n\nPair {pair_id+1}, Accuracy: {acc}, Similarity Proximity: {sim_prox}")

                matching_accs.append(acc)
                sim_proxs.append(sim_prox)
                best_epochs.append(best_epoch)
                comp_times.append(elapsed_time)

            # Average metrics
            avg_acc = np.mean(matching_accs)
            std_acc = np.std(matching_accs)
            avg_sim_prox = np.mean(sim_proxs)
            std_sim_prox = np.std(sim_proxs)
            avg_time = np.mean(comp_times)
            avg_best_epoch = np.mean(best_epochs)

            # Write results
            out_data = [{
                'model': model_name,
                'data': data_name,
                'size': len(dataset),
                'p_add': p_add,
                'p_rm': p_rm,
                'avg_acc': avg_acc,
                'std_acc': std_acc,
                'avg_sim_prox': avg_sim_prox,
                'std_sim_prox': std_sim_prox, 
                'avg_time': avg_time,
                'avg_best_epoch': avg_best_epoch,
            }]

            with open(res_file, 'a', newline='') as rf:
                csv_writer = csv.DictWriter(rf, fieldnames=header)
                csv_writer.writerows(out_data)

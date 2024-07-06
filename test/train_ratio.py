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
from netalign.evaluation.metrics import compute_accuracy, compute_conf_score
from netalign.models import init_align_model


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description='Read the test configuration.')

    parser.add_argument('-c', '--cfg', type=str, required=True, help="Path to the model configuration (YAML file).")
    parser.add_argument('-s', '--source_path', type=str, required=True, help="Path to the source network in edgelist format.")
    parser.add_argument('--size', type=int, default=10, help="Number of random target network copies. Default: 10")
    parser.add_argument('--noise', type=float, default=0.05, help="Percentage of noise in addition of the target network. Default: 0.05")
    parser.add_argument('--seed', type=int, default=None, help="Seed for reproducibility. Default: None")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for training. Default: cuda")
    parser.add_argument('--res_dir', type=str, default='results/train_ratio', help="Path to the directory where to save the test results. Default: results/train_ratio")
    
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

    res_file = f'{args.res_dir}_noise_{args.noise}/{model_name}_{data_name}.csv'

    header = ['model', 'data', 'size', 'train_ratio', 'p_add', 'p_rm', 'avg_acc', 'std_acc', 'avg_conf_score', 'std_conf_score', 'avg_time', 'avg_best_epoch']
    with open(res_file, 'w') as rf:
        csv_writer = csv.DictWriter(rf, fieldnames=header)
        csv_writer.writeheader()
    
    # Run tests
    noise_types = ['add']
    noise_probs = [args.noise]
    train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for noise_type in noise_types:
        for noise_prob in noise_probs:
            for train_ratio in train_ratios:
                p_add = noise_prob if noise_type == 'add' else 0.0
                p_rm = noise_prob if noise_type == 'rm' else 0.0

                dataset = SemiSyntheticDataset(
                    source_path=args.source_path,
                    size=args.size,
                    permute=True,
                    p_add=p_add,
                    p_rm=p_rm,
                    train_ratio=train_ratio,
                    val_ratio=0,
                    seed=args.seed
                )
                
                dataloader = DataLoader(dataset, shuffle=True)

                matching_accs = []
                conf_scores = []
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
                    try:
                        S, best_epoch = model.align(pair_dict)
                    except: # It may happen that for low train ratio a model can't conclude alignment
                        S = np.zeros((pair_dict['graph_pair'][0].num_nodes, pair_dict['graph_pair'][1].num_nodes))
                        best_epoch = -1
                    elapsed_time = time.time() - start_time

                    # Compute matching accuracy
                    P = greedy_match(S)
                    gt_test = dict_to_perm_mat(pair_dict['gt_test'], pair_dict['graph_pair'][0].num_nodes, pair_dict['graph_pair'][1].num_nodes).detach().cpu().numpy()
                    acc = compute_accuracy(P, gt_test)

                    # Compute similarity proximity score
                    conf_score = compute_conf_score(S)

                    print(f"\n\nPair {pair_id+1}, Accuracy: {acc}, Confidence Score: {conf_score}")

                    matching_accs.append(acc)
                    conf_scores.append(conf_score)
                    best_epochs.append(best_epoch)
                    comp_times.append(elapsed_time)

                # Average metrics
                avg_acc = np.mean(matching_accs)
                std_acc = np.std(matching_accs)
                avg_conf_score = np.mean(conf_scores)
                std_conf_score = np.std(conf_scores)
                avg_time = np.mean(comp_times)
                avg_best_epoch = np.mean(best_epochs)

                # Write results
                out_data = [{
                    'model': model_name,
                    'data': data_name,
                    'size': len(dataset),
                    'train_ratio': train_ratio,
                    'p_add': p_add,
                    'p_rm': p_rm,
                    'avg_acc': avg_acc,
                    'std_acc': std_acc,
                    'avg_conf_score': avg_conf_score,
                    'std_conf_score': std_conf_score, 
                    'avg_time': avg_time,
                    'avg_best_epoch': avg_best_epoch,
                }]

                with open(res_file, 'a', newline='') as rf:
                    csv_writer = csv.DictWriter(rf, fieldnames=header)
                    csv_writer.writerows(out_data)

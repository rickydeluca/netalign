import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from netalign.data.dataset import SemiSyntheticDataset
from netalign.models import init_align_model


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description='Read the test configuration.')

    parser.add_argument('-c', '--cfg', type=str, required=True, help="Path to the model configuration (YAML file).")
    parser.add_argument('-s', '--source_path', type=str, required=True, help="Path to the source network in edgelist format.")
    parser.add_argument('--size', type=int, default=100, help="Number of random target network copies. Default: 10")
    parser.add_argument('--train_ratio', type=float, default=0.2, help="Percentage of nodes/graphs to use as training set. Default: 0.2")
    parser.add_argument('--val_ratio', type=float, default=0.0, help="Percentage of nodes/graphs to use as validation set. Default: 0.0")
    parser.add_argument('--seed', type=int, default=None, help="Seed for reproducibility. Default: None")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for training. Default: cuda")
    parser.add_argument('--res_dir', type=str, default='results/split_on_graphs', help="Path to the directory where to save the test results. Default: results/split_on_graphs")
    
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

    # Init model, once and for all!
    model, model_name = init_align_model(cfg)
    model.to(device)

    res_file = f'{args.res_dir}/{model_name}_{data_name}.csv'

    header = ['model', 'data', 'size', 'p_add', 'p_rm', 'avg_acc', 'std_acc', 'training_time']
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

            # Prepare datasets
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

            train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])

            # Dataloaders
            train_loader = DataLoader(train_dataset, shuffle=True)
            val_loader = DataLoader(val_dataset, shuffle=True)
            test_loader = DataLoader(test_dataset, shuffle=False)

            # Train model
            start_time = time.time()
            model.train_eval(train_loader, val_loader=val_loader, device=device, verbose=False)
            training_time = time.time() - start_time

            # Test
            avg_acc, std_acc = model.evaluate(test_loader, use_acc=True)

            # Write results
            out_data = [{
                'model': model_name,
                'data': data_name,
                'size': len(dataset),
                'p_add': p_add,
                'p_rm': p_rm,
                'avg_acc': avg_acc,
                'std_acc': std_acc,
                'training_time': training_time
            }]

            with open(res_file, 'a', newline='') as rf:
                csv_writer = csv.DictWriter(rf, fieldnames=header)
                csv_writer.writerows(out_data)

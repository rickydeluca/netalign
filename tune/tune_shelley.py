import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from netalign.data.dataset import SemiSyntheticDataset
from netalign.evaluation.kfold import k_fold_cv

# param_grid = {'feat': ['share', 'degree'],
#               'embedding': ['gine2', 'gine', 'gcn'],
#               'matching': ['sgm', 'sigma'],
#               'hidden_channels': [32, 64, 256, 512, 1024],
#               'num_layers': [1, 2, 3, 4, 5],
#               'tau': [0.001, 0.01, 0.1, 1],
#               'beta': [0.0001, 0.001, 0.01, 0.1],
#               'epochs': [10, 20, 50, 100, 500],
#               'lr': [1e-4, 1e-3, 1e-2, 1e-1],
#               'optimizer': ['adam', 'sgd'],
#               'weight_decay': [0, 1e-4, 1e-3],
#               'momentum': [0.99, 0.9, 0.5]}

param_grid = {'feat': ['share'],
              'embedding': ['gine2'],
              'matching': ['sgm'],
              'hidden_channels': [256,512,1024],
              'num_layers': [2,3,4],
              'tau': [1],
              'beta': [0.1],
              'epochs': [30, 100],
              'lr': [1e-4, 1e-3],
              'optimizer': ['adam'],
              'weight_decay': [1e-4],
              'momentum': [0.99, 0.9, 0.5]}

total_iterations = (
    len(param_grid['feat']) *
    len(param_grid['embedding']) *
    len(param_grid['hidden_channels']) *
    len(param_grid['num_layers']) *
    len(param_grid['matching']) *
    len(param_grid['tau']) *
    len(param_grid['beta']) *
    len(param_grid['optimizer']) *
    len(param_grid['epochs']) *
    len(param_grid['lr']) *
    len(param_grid['weight_decay'])      # Same as momentum
)

def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description='Read the test configuration.')

    parser.add_argument('-s', '--source_path', type=str, required=True, help="Path to the source network in edgelist format.")
    parser.add_argument('--cv', type=int, default=5, help="Number of folds for k-fold cross validation. Default: 5")
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility. Default: 42")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for training. Default: cuda")
    parser.add_argument('--out_dir', type=str, default='output/best_cfgs', help="Path to the directory where to save the best configuration. Default: output/best_cfgs")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Init output file
    data_name = args.source_path.split('/')[-1].split('.')[0]
    csv_path = f"{args.out_dir}/shelley_{data_name}.csv"
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['avg_acc', 'cfg'])
    
    # Reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    
    # Create only one pair, the tuning will be done on this
    # using a k-fold cross validation strategy
    dataset = SemiSyntheticDataset(
        source_path=args.source_path,
        size=1,
        permute=True,
        p_add=0.0,
        p_rm=0.2,
        train_ratio=1.0,    # The whole groundtruth is in `pair_dict['gt_train']`
        seed=args.seed
    )

    dataloder = DataLoader(dataset)
    pair_dict = next(iter(dataloder))

    progress_bar = tqdm(total=total_iterations, desc="Progress")    

    cfg = edict()
    cfg.NAME = 'shelley'
    for feat in param_grid['feat']:
        cfg.FEATS = edict()
        cfg.FEATS.TYPE = feat
        cfg.FEATS.FEATURE_DIM = 1
        
        for embedding in param_grid['embedding']:
            cfg.EMBEDDING = edict()
            cfg.EMBEDDING.MODEL = embedding
            
            for hidden_channels in param_grid['hidden_channels']:
                for num_layers in param_grid['num_layers']:
                    
                    if embedding == 'gine' or embedding == 'gine2':
                        cfg.EMBEDDING.IN_CHANNELS = 1
                        cfg.EMBEDDING.DIM = hidden_channels
                        cfg.EMBEDDING.OUT_CHANNELS = hidden_channels
                        cfg.EMBEDDING.NUM_CONV_LAYERS = num_layers
                    else:
                        cfg.EMBEDDING.IN_CHANNELS = 1
                        cfg.EMBEDDING.HIDDEN_CHANNELS = hidden_channels
                        cfg.EMBEDDING.OUT_CHANNELS = hidden_channels * 2
                        cfg.EMBEDDING.NUM_LAYERS = num_layers

                    for matching in param_grid['matching']:
                        cfg.MATCHING = edict()
                        cfg.MATCHING.MODEL = matching
                        
                        for tau in param_grid['tau']:
                            for beta in param_grid['beta']:
                                if matching == 'sigma':
                                    cfg.MATCHING.T = 5
                                    cfg.MATCHING.MISS_MATCH_VALUE = beta
                                    cfg.MATCHING.TAU = tau
                                    cfg.MATCHING.N_SINK_ITERS = 10
                                    cfg.MATCHING.N_SAMPLES = 10
                                if matching == 'sgm':
                                    cfg.MATCHING.N_SINK_ITERS = 10
                                    cfg.MATCHING.BETA = beta
                                    cfg.MATCHING.TAU = tau
                                    cfg.MATCHING.MASK = True

                                for optimizer in param_grid['optimizer']:
                                    cfg.TRAIN = edict()
                                    cfg.TRAIN.OPTIMIZER = optimizer

                                    for epochs in param_grid['epochs']:
                                        for lr in param_grid['lr']:
                                            cfg.TRAIN.EPOCHS = epochs
                                            cfg.TRAIN.LR = lr
                                            cfg.TRAIN.PATIENCE = epochs

                                            if optimizer == 'adam':
                                                weight_decays = param_grid['weight_decay']
                                                momentums = [None]
                                            if optimizer == 'sgd':
                                                weight_decays = [None]
                                                momentums = param_grid['momentum']
                                            
                                            for weight_decay in weight_decays:
                                                for momentum in momentums:
                                                    cfg.TRAIN.MOMENTUM = momentum
                                                    cfg.TRAIN.L2NORM = weight_decay

                                                    # Run k-fold cross validation
                                                    acc = k_fold_cv(cfg, dataset, device=device, seed=args.seed)

                                                    # Write to CSV file
                                                    with open(csv_path, 'a', newline='') as csvfile:
                                                        writer = csv.writer(csvfile)
                                                        writer.writerow([acc, cfg])

                                                    # Update progress
                                                    progress_bar.update(1)
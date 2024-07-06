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

# param_grid = {'emb_dim': [32, 64, 256, 512, 1024],
#               'emb_lr': [1e-4, 1e-3, 1e-2],
#               'emb_epochs': [10, 100, 1000],
#               'num_hidden': [0, 1, 2, 3],
#               'loss_funcs': ['eculidean', 'contrastive'],
#               'map_epochs': [10, 50, 100],
#               'map_lr': [1e-4, 1e-3, 1e-2],
#               'optimizer': ['adam'],
#               'weight_decay': [0, 1e-4, 1e-3]}

param_grid = {'emb_dim': [32, 64, 256, 512, 1024],
              'emb_lr': [1e-4, 1e-3, 1e-2],
              'emb_epochs': [10, 100, 1000],
              'num_hidden': [0, 1, 2, 3],
              'loss_funcs': ['eculidean'],
              'map_epochs': [10, 50, 100],
              'map_lr': [1e-4, 1e-3, 1e-2],
              'optimizer': ['adam'],
              'weight_decay': [0, 1e-4, 1e-3]}

total_iterations = (
    len(param_grid['emb_dim']) *
    len(param_grid['emb_lr']) *
    len(param_grid['emb_epochs']) *
    len(param_grid['num_hidden']) *
    len(param_grid['loss_funcs']) *
    len(param_grid['map_epochs']) *
    len(param_grid['map_lr']) *
    len(param_grid['optimizer']) *
    len(param_grid['weight_decay'])
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
    csv_path = f"{args.out_dir}/pale_{data_name}.csv"
    
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
        p_add=0.2,
        p_rm=0.2,
        train_ratio=1.0,    # The whole groundtruth is in `pair_dict['gt_train']`
        seed=args.seed
    )

    dataloder = DataLoader(dataset)
    pair_dict = next(iter(dataloder))

    progress_bar = tqdm(total=total_iterations, desc="Progress")    

    cfg = edict()
    cfg.NAME = 'pale'
    cfg.DEVICE = args.device
    for emb_dim in param_grid['emb_dim']:
        for emb_epochs in param_grid['emb_epochs']:
            for emb_lr in param_grid['emb_lr']:
                cfg.EMBEDDING = edict()
                cfg.EMBEDDING.NEG_SAMPLE_SIZE = 10
                cfg.EMBEDDING.EMBEDDING_DIM = emb_dim
                cfg.EMBEDDING.EMBEDDING_NAME = ''
                cfg.EMBEDDING.OPTIMIZER = 'Adam'
                cfg.EMBEDDING.LR = emb_lr
                cfg.EMBEDDING.EPOCHS = emb_epochs

                for num_hidden in param_grid['num_hidden']:
                    for loss_fn in param_grid['loss_funcs']:
                        for map_epochs in param_grid['map_epochs']:
                            for map_lr in param_grid['map_lr']:
                                for l2norm in param_grid['weight_decay']:
                                    cfg.MAPPING.NUM_HIDDEN = num_hidden
                                    cfg.MAPPING.ACTIVATE_FUNCTION = 'sigmoid'
                                    cfg.MAPPING.LOSS_FUNCTION = loss_fn
                                    cfg.MAPPING.OPTIMIZER = 'Adam'
                                    cfg.MAPPING.LR = map_lr
                                    cfg.MAPPING.BATCH_SIZE_TRAIN = 8 if data_name == 'edi3' else 32
                                    cfg.MAPPING.BATCH_SIZE_VAL = 8 if data_name == 'edi3' else 32
                                    cfg.MAPPING.EPOCHS = map_epochs
                                    cfg.MAPPING.PATIENCE = map_epochs
                                    cfg.MAPPING.VALIDATE = False

                                    # Run k-fold cross validation
                                    acc = k_fold_cv(cfg, dataset, device=device, seed=args.seed)

                                    # Write to CSV file
                                    with open(csv_path, 'a', newline='') as csvfile:
                                        writer = csv.writer(csvfile)
                                        writer.writerow([acc, json.dumps(cfg)])

                                    # Update progress
                                    progress_bar.update(1)



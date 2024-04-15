import csv
import os
import time

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from netalign.data.dataset import RobustnessDataset
from netalign.models.shelley.shelley import SHELLEY_G

if __name__ == '__main__':

    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    feats = ['share', 'degree']
    embedding_models = ['gine2', 'gine', 'gcn']
    embedding_dims = [512]
    num_layers = [4]
    matching_models = ['sgm', 'linear']


    # Training
    epochs = 1000
    patience = 20
    # learning_rates = [0.01, 0.001, 0.0001]
    learning_rates = [0.0001]


    # Data
    data = 'bn1000'
    subset = ['ad']
    alteration_modes = ['add', 'remove']
    noise_levels = [0.00, 0.05, 0.10, 0.20, 0.25]

    # Output directory
    res_dir = 'results/robustness/graph_split/shelley'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    for feat in feats:
        for embedding_model in embedding_models:
            for matching_model in matching_models:
                betas = [0.1] if matching_model == 'sgm' else []
                taus = [1.0] if matching_model == 'sgm' else []
                for beta in betas:
                    for tau in taus:
                        for ed in embedding_dims:
                            for nl in num_layers:
                                for lr in learning_rates:
                                    # Outfile
                                    if matching_model == 'sgm':
                                        model_name = f"{feat}-{embedding_model}-{matching_model}-beta{beta}-tau{tau}"
                                    else:
                                        model_name = f"{feat}-{embedding_model}-{matching_model}"

                                    csv_path = f'{res_dir}/{model_name}_ed{ed}_nl{nl}_lr{lr}.csv'
                                    
                                    header = ['model', 'data', 'p_add', 'p_rm', 'dataset_size', 'best_val_acc', 'best_epoch', 'test_acc', 'training_time']
                                    with open(csv_path, 'w') as outfile:
                                        csv_writer = csv.DictWriter(outfile, fieldnames=header)
                                        csv_writer.writeheader()

                                    # Configuration dictionary
                                    cfg = edict()

                                    cfg.SEED = seed
                                    cfg.DEVICE = device

                                    cfg.DATA = edict()
                                    cfg.DATA.NAME = data
                                    cfg.DATA.PATH = f'data/{data}'
                                    cfg.DATA.SUBSET = subset
                                    cfg.DATA.TRAIN_RATIO = 0.15 # Unused
                                    cfg.DATA.VAL_RATIO = 0.15   # Unused
                                    cfg.DATA.NUM_COPIES = 25
                                    cfg.DATA.GT_MODE = 'matrix'

                                    cfg.MODEL = edict()
                                    cfg.MODEL.NAME = 'shelley'

                                    cfg.INIT = edict()
                                    cfg.INIT.FEATURES = feat
                                    cfg.INIT.FEATURE_DIM = 1

                                    cfg.EMBEDDING = edict()
                                    cfg.EMBEDDING.MODEL = embedding_model
                                    
                                    if embedding_model == 'gcn':
                                        cfg.EMBEDDING.IN_CHANNELS = 1
                                        cfg.EMBEDDING.HIDDEN_CHANNELS = ed * 2
                                        cfg.EMBEDDING.OUT_CHANNELS = ed
                                        cfg.EMBEDDING.NUM_LAYERS = nl
                                    else:
                                        cfg.EMBEDDING.IN_CHANNELS = 1
                                        cfg.EMBEDDING.DIM = ed
                                        cfg.EMBEDDING.OUT_CHANNELS = ed
                                        cfg.EMBEDDING.NUM_CONV_LAYERS = nl

                                    cfg.MATCHING = edict()
                                    cfg.MATCHING.MODEL = matching_model

                                    if matching_model == 'sgm':
                                        cfg.MATCHING.N_SINK_ITERS = 10
                                        cfg.MATCHING.BETA = beta
                                        cfg.MATCHING.TAU = tau
                                    elif matching_model == 'linear':
                                        if embedding_model == 'gine':
                                            cfg.MATCHING.EMBEDDING_DIM = ed * nl
                                        else:
                                            cfg.MATCHING.EMBEDDING_DIM = ed
                                    else:
                                        raise ValueError("Invalid matching model!")


                                    cfg.TRAIN = edict()
                                    cfg.TRAIN.OPTIMIZER = 'adam'
                                    cfg.TRAIN.LR = lr
                                    cfg.TRAIN.L2NORM = 0.0
                                    cfg.TRAIN.EPOCHS = epochs
                                    cfg.TRAIN.PATIENCE = patience

                                    for alteration_mode in alteration_modes:
                                        for noise_level in noise_levels:

                                            if alteration_mode == 'add':
                                                p_add = noise_level
                                                p_rm = 0.0
                                            elif alteration_mode == 'remove':
                                                p_add = 0.0
                                                p_rm = noise_level
                                            else:
                                                raise ValueError("Invalid alteration mode!")
                                            
                                            # Init model
                                            model = SHELLEY_G(cfg).to(device)

                                            # Dataset
                                            dataset = RobustnessDataset(
                                                root_dir=cfg.DATA.PATH,
                                                subset=cfg.DATA.SUBSET,
                                                train_ratio=cfg.DATA.TRAIN_RATIO,
                                                val_ratio=cfg.DATA.VAL_RATIO,
                                                seed=cfg.SEED,
                                                p_add=p_add,
                                                p_rm=p_rm,
                                                num_copies=cfg.DATA.NUM_COPIES,
                                                gt_mode=cfg.DATA.GT_MODE
                                            )

                                            # Train, val and test split
                                            train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])

                                            # Dataloaders
                                            train_loader = DataLoader(train_dataset, shuffle=True)
                                            val_loader = DataLoader(val_dataset, shuffle=True)
                                            test_loader = DataLoader(test_dataset, shuffle=False)
                                            
                                            # Train model
                                            start_time = time.time()
                                            best_val_acc, best_epoch = model.train_eval(train_loader, val_loader)
                                            training_time = time.time() - start_time
                                            
                                            # Test
                                            test_acc = model.evaluate(test_loader, use_acc=True)

                                            # Save output data
                                            output_data = [{
                                                'model': model_name,
                                                'data': cfg.DATA.NAME,
                                                'p_add': p_add,
                                                'p_rm': p_rm,
                                                'dataset_size': len(dataset),
                                                'best_val_acc': best_val_acc,
                                                'best_epoch': best_epoch,
                                                'test_acc': test_acc,
                                                'training_time': training_time
                                            }]

                                            # Write result to CSV
                                            with open(csv_path, 'a', newline='') as of:
                                                csv_writer = csv.DictWriter(of, fieldnames=header)
                                                csv_writer.writerows(output_data)

                                            if p_add == 0.0 and p_rm == 0.0 and test_acc < 0.1:
                                                break
import csv
import os
import time

import numpy as np
import torch
from easydict import EasyDict as edict
from netalign.data.utils import move_tensors_to_device
from netalign.evaluation.matchers import greedy_match
from netalign.evaluation.metrics import compute_accuracy
from torch_geometric.loader import DataLoader

from netalign.data.dataset import RobustnessDataset
from netalign.models.shelley.shelley import SHELLEY_N

if __name__ == '__main__':

    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    feats = ['share']
    embedding_models = ['gine2', 'gine', 'gcn']
    embedding_dims = [32, 512]
    num_layers = [2]
    matching_models = ['sgm']


    # Training
    epochs = 1000
    patience = 100
    learning_rates = [0.00001]


    # Data
    data = 'edi3'
    subset = ['edi3']
    alteration_modes = ['add', 'remove']
    noise_levels = [0.00, 0.05, 0.10, 0.20, 0.25]

    # Output directory
    res_dir = f'results/robustness/shelley/{data}'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    for feat in feats:
        for embedding_model in embedding_models:
            for matching_model in matching_models:
                betas = [0.001] if matching_model == 'sgm' else ['']
                taus = [1.0] if matching_model == 'sgm' else ['']
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

                                    csv_path = f'{res_dir}/{data}_{model_name}_ed{ed}_nl{nl}_lr{lr}.csv'
                                    
                                    header = ['model', 'data', 'p_add', 'p_rm', 'dataset_size', 'avg_acc', 'std_dev', 'avg_time', 'avg_best_epoch']
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
                                    cfg.DATA.TRAIN_RATIO = 0.15
                                    cfg.DATA.VAL_RATIO = 0.15
                                    cfg.DATA.NUM_COPIES = 5
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
                                        cfg.MATCHING.MASK = True
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

                                            dataloader = DataLoader(dataset, shuffle=True)

                                            matching_accuracies = []
                                            comp_times = []
                                            best_epochs = []

                                            for i, pair_dict in enumerate(dataloader):
                                                # Init model
                                                model = SHELLEY_N(cfg).to(device)

                                                # Move data to device
                                                pair_dict = move_tensors_to_device(pair_dict, device)

                                                # Align networks
                                                start_time = time.time()

                                                S, best_epoch = model.align(pair_dict)

                                                elapsed_time = time.time() - start_time

                                                # Compute accuracy
                                                test_gt = (pair_dict['gt_matrix'] * pair_dict['test_mask']).squeeze(0).detach().cpu().numpy()
                                                pred_mat = greedy_match(S)

                                                acc = compute_accuracy(pred_mat, test_gt)
                                                
                                                print(f"Pair {i}, accuracy: ", acc.item())
                                                
                                                matching_accuracies.append(acc.item())
                                                comp_times.append(elapsed_time)
                                                best_epochs.append(best_epoch)

                                            # Average computation time
                                            avg_time = np.mean(comp_times)

                                            # Average accuracy and std deviation
                                            avg_accuracy = np.mean(matching_accuracies)
                                            std_deviation = np.std(matching_accuracies)

                                            # Average best epoch
                                            avg_best_epoch = np.mean(best_epochs)

                                            # Save output data
                                            output_data = [{
                                                'model': f'{cfg.INIT.FEATURES}-{cfg.EMBEDDING.MODEL}-{cfg.MATCHING.MODEL}',
                                                'data': cfg.DATA.NAME,
                                                'p_add': p_add,
                                                'p_rm': p_rm,
                                                'dataset_size': len(dataset),
                                                'avg_acc': avg_accuracy,
                                                'std_dev': std_deviation,
                                                'avg_time': avg_time,
                                                'avg_best_epoch': avg_best_epoch
                                            }]

                                            print('Average matching accuracy: ', avg_accuracy)
                                            print('Std deviation: ', std_deviation)

                                            # Write result to CSV
                                            with open(csv_path, 'a', newline='') as of:
                                                csv_writer = csv.DictWriter(of, fieldnames=header)
                                                csv_writer.writerows(output_data)

                                            if p_add == 0.0 and p_rm == 0.0 and avg_accuracy < 0.1:
                                                break
import csv
import os
import time

import numpy as np
import torch
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader

from align import align_networks
from netalign.data.dataset import RobustnessDataset
from netalign.evaluation.dict_to_mat import dict_to_perm_mat
from netalign.evaluation.metrics import compute_accuracy
from netalign.evaluation.matcher import greedy_match

if __name__ == '__main__':

    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define configurations
    data = ['edi3']
    validate = [False]
    map_modes = ['double', 'single']
    alteration_modes = ['add', 'remove']
    loss_fns = ['euclidean', 'contrastive']
    hidden_layers = [0, 1]
    map_lrs = [0.01, 0.001]
    noise_levels = [0.00, 0.05, 0.10, 0.20, 0.25]

    res_dir = 'results/robustness/pale'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    for d in data:
        for val in validate:
            for map_mode in map_modes:
                for hl in hidden_layers:
                    for lr in map_lrs:
                        for loss_fn in loss_fns:
                            # Define model name
                            mm = 's' if map_mode == 'single' else 'd'
                            lf = 'e' if loss_fn == 'euclidean' else 'c'
                            model_name = f'pale-mlp{hl}-{mm}{lf}_lr{lr}'
                            # Outfile
                            if val is True:
                                csv_path = f'{res_dir}/{d}_{model_name}_val.csv'
                            else:
                                csv_path = f'{res_dir}/{d}_{model_name}.csv'

                            header = ['model', 'data', 'p_add', 'p_rm', 'dataset_size', 'avg_acc', 'std_dev', 'avg_time', 'avg_best_epoch']
                            with open(csv_path, 'w') as outfile:
                                csv_writer = csv.DictWriter(outfile, fieldnames=header)
                                csv_writer.writeheader()

                            # Configuration dictionary
                            cfg = edict()

                            cfg.SOLVER = 'greedy'
                            cfg.SEED = seed
                            cfg.DEVICE = 'cuda'

                            cfg.DATA = edict()
                            cfg.DATA.PATH = f'data/{d}'
                            cfg.DATA.SIZE = 3

                            if val is True:
                                cfg.DATA.TRAIN_RATIO = 0.15
                                cfg.DATA.VAL_RATIO = 0.15
                            else:
                                cfg.DATA.TRAIN_RATIO = 0.2
                                cfg.DATA.VAL_RATIO = None

                            cfg.MODEL = edict()
                            cfg.MODEL.NAME = model_name

                            cfg.MODEL.EMBEDDING = edict()
                            cfg.MODEL.EMBEDDING.NEG_SAMPLE_SIZE = 10
                            cfg.MODEL.EMBEDDING.EMBEDDING_DIM = 300
                            cfg.MODEL.EMBEDDING.EMBEDDING_NAME = ''

                            cfg.MODEL.EMBEDDING.OPTIMIZER = 'Adam'
                            cfg.MODEL.EMBEDDING.LR = 0.01
                            cfg.MODEL.EMBEDDING.BATCH_SIZE = 512
                            cfg.MODEL.EMBEDDING.EPOCHS = 1000

                            cfg.MODEL.MAPPING = edict()
                            cfg.MODEL.MAPPING.MODE = map_mode
                            cfg.MODEL.MAPPING.LOSS_FUNCTION = loss_fn
                            cfg.MODEL.MAPPING.N_SINK_ITERS = 10
                            cfg.MODEL.MAPPING.TAU = 1.0
                            cfg.MODEL.MAPPING.BETA = 0.001
                            cfg.MODEL.MAPPING.TOP_K = 5
                            cfg.MODEL.MAPPING.NUM_HIDDEN = hl
                            cfg.MODEL.MAPPING.ACTIVATE_FUNCTION = 'sigmoid'
                            cfg.MODEL.MAPPING.OPTIMIZER = 'Adam'
                            cfg.MODEL.MAPPING.LR = lr
                            cfg.MODEL.MAPPING.VALIDATE = val

                            if val is True:
                                cfg.MODEL.MAPPING.VALIDATE = True
                                cfg.MODEL.MAPPING.EPOCHS = 1000
                                cfg.MODEL.MAPPING.PATIENCE = 100
                                cfg.MODEL.MAPPING.BATCH_SIZE_TRAIN = 8
                                cfg.MODEL.MAPPING.BATCH_SIZE_VAL = 8
                            else:
                                cfg.MODEL.MAPPING.VALIDATE = False
                                cfg.MODEL.MAPPING.EPOCHS = 100
                                cfg.MODEL.MAPPING.PATIENCE = 100
                                cfg.MODEL.MAPPING.BATCH_SIZE_TRAIN = 8
                                cfg.MODEL.MAPPING.BATCH_SIZE_VAL = None

                            for alteration_mode in alteration_modes:
                                for noise_level in noise_levels:

                                    if alteration_mode == 'add':
                                        p_add = noise_level
                                        p_rm = 0
                                    elif alteration_mode == 'remove':
                                        p_add = 0
                                        p_rm = noise_level
                                    else:
                                        raise ValueError("Invalid alteration mode!")
                                    
                                    # Dataset
                                    pair_dataset = RobustnessDataset(
                                        root_dir=cfg.DATA.PATH,
                                        subset=['edi3'],
                                        p_add=p_add,
                                        p_rm=p_rm,
                                        num_copies=cfg.DATA.SIZE,
                                        train_ratio=cfg.DATA.TRAIN_RATIO,
                                        val_ratio=cfg.DATA.VAL_RATIO,
                                        seed=cfg.SEED,
                                        gt_mode='dictionary')
                                    
                                    dataloader = DataLoader(pair_dataset)

                                    # --- Train & Evaluate ---
                                    matching_accuracies = []
                                    comp_times = []
                                    best_epochs = []

                                    for i, pair_dict in enumerate(dataloader):
                                        # Get alignment matrix
                                        start_time = time.time()
                                        sim_matrix, best_epoch = align_networks(pair_dict, cfg)
                                        end_time = time.time()
                                    
                                        # Get groundtruth permutation matrix
                                        num_source_nodes = pair_dict['graph_pair'][0].num_nodes
                                        num_target_nodes = pair_dict['graph_pair'][0].num_nodes
                                        groundtruth = dict_to_perm_mat(
                                            pair_dict['test_dict'],
                                            n_sources=num_source_nodes,
                                            n_targets=num_target_nodes
                                        )
                                        
                                        pred_mat = greedy_match(sim_matrix)

                                        acc = compute_accuracy(
                                            pred_mat,
                                            groundtruth
                                        )
                                        
                                        print(f"Pair {i}, accuracy: ", acc.item())
                                        
                                        matching_accuracies.append(acc.item())
                                        comp_times.append(end_time - start_time)
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
                                        'model': cfg.MODEL.NAME,
                                        'data': d,
                                        'p_add': p_add,
                                        'p_rm': p_rm,
                                        'dataset_size': len(pair_dataset),
                                        'avg_acc': avg_accuracy,
                                        'std_dev': std_deviation,
                                        'avg_time': avg_time,
                                        'avg_best_epoch': avg_best_epoch
                                    }]

                                    print('Average matching accuracy: ', avg_accuracy)
                                    print('Std deviation: ', std_deviation)

                                    # Write result to CSV
                                    with open(csv_path, 'a', newline='') as outfile:
                                        csv_writer = csv.DictWriter(outfile, fieldnames=header)
                                        csv_writer.writerows(output_data)
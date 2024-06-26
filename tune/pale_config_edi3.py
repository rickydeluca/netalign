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
from netalign.evaluation.matchers import greedy_match

if __name__ == '__main__':

    # Reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Possible configs
    model_name = 'pale'
    data = 'edi3'
    subset = 'edi3'
    map_modes = ['single', 'double']
    hidden_layers = [0, 1, 2]
    loss_functions = ['euclidean', 'contrastive']
    learning_rates = [0.01, 0.001, 0.001]

    # Output dir
    res_dir = 'results/configs/pale'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    for lr in learning_rates:
        for mm in map_modes:
            for hl in hidden_layers:
                for lf in loss_functions:
                
                        # Init output file
                        outfile = f'{res_dir}/pale_{data}_configs.csv'
                        header = ['data', 'learning_rate', 'map_mode', 'num_hidden_layers', 'loss_function', 'avg_acc', 'std_dev', 'avg_time', 'avg_best_epoch']
                        with open(outfile, 'w') as of:
                            csv_writer = csv.DictWriter(of, fieldnames=header)
                            csv_writer.writeheader()

                        # Configuration dictionary
                        cfg = edict()

                        cfg.SOLVER = 'greedy'
                        cfg.SEED = seed
                        cfg.DEVICE = device

                        cfg.DATA = edict()
                        cfg.DATA.NAME = data
                        cfg.DATA.SUBSET = subset
                        cfg.DATA.PATH = f'data/{data}'
                        cfg.DATA.NUM_COPIES = 5
                        cfg.DATA.TRAIN_RATIO = 0.15
                        cfg.DATA.VAL_RATIO = 0.15

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
                        cfg.MODEL.MAPPING.MODE = mm
                        cfg.MODEL.MAPPING.LOSS_FUNCTION = lf
                        cfg.MODEL.MAPPING.N_SINK_ITERS = 10
                        cfg.MODEL.MAPPING.TAU = 1.0
                        cfg.MODEL.MAPPING.BETA = 0.001
                        cfg.MODEL.MAPPING.NUM_HIDDEN = hl
                        cfg.MODEL.MAPPING.ACTIVATE_FUNCTION = 'sigmoid'
                        cfg.MODEL.MAPPING.OPTIMIZER = 'Adam'
                        cfg.MODEL.MAPPING.LR = lr
                        cfg.MODEL.MAPPING.VALIDATE = True
                        cfg.MODEL.MAPPING.TOP_K = None

                        cfg.MODEL.MAPPING.VALIDATE = True
                        cfg.MODEL.MAPPING.EPOCHS = 1000
                        cfg.MODEL.MAPPING.PATIENCE = 20
                        cfg.MODEL.MAPPING.BATCH_SIZE_TRAIN = 8
                        cfg.MODEL.MAPPING.BATCH_SIZE_VAL = 8

                        # Dataset
                        pair_dataset = RobustnessDataset(
                            root_dir=cfg.DATA.PATH,
                            subset=cfg.DATA.SUBSET,
                            p_add=0.2,
                            p_rm=0.2,
                            num_copies=cfg.DATA.NUM_COPIES,
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
                            'data': data,
                            'learning_rate': lr,
                            'dataset_size': len(pair_dataset),
                            'avg_acc': avg_accuracy,
                            'std_dev': std_deviation,
                            'avg_time': avg_time,
                            'avg_best_epoch': avg_best_epoch
                        }]

                        print('Average matching accuracy: ', avg_accuracy)
                        print('Std deviation: ', std_deviation)

                        # Write result to CSV
                        with open(outfile, 'a', newline='') as of:
                            csv_writer = csv.DictWriter(of, fieldnames=header)
                            csv_writer.writerows(output_data)
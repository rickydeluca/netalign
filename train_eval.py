import argparse
import csv
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader

from align import align_networks
from netalign.data.dataset import SemiSyntheticDataset
from netalign.evaluation.metrics import compute_accuracy


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description='Read the configuration file for the model training.'
    )

    parser.add_argument('-c', '--config', type=str, required=True,
                        help="Path to yaml configuration file.")
    
    return parser.parse_args()


def read_config_file(yaml_file):
    """
    Read the yaml configuration file and return it
    as dictionary.
    """
    with open(yaml_file) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    return cfg


if __name__ == '__main__':

    # Read configuration file
    args = parse_args()
    cfg = read_config_file(args.config)

    # Set default device
    if cfg.DEVICE == 'cuda':
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Reproducibility
    torch.manual_seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)

    # Init output CSV file
    data_name = cfg.DATA.PATH.replace('data/', '')
    csv_path = f'results/{cfg.MODEL.NAME.lower()}_{cfg.MATCHER}_{data_name}_v{cfg.VERSION}.csv'
    header = ['model', 'data', 'p_add', 'p_rm', 'num_tests', 'avg_acc', 'std_dev', 'avg_time']
    with open(csv_path, 'w') as outfile:
        csv_writer = csv.DictWriter(outfile, fieldnames=header)
        csv_writer.writeheader()

    # Iterate over data configurations
    p_adds = cfg.DATA.P_ADD
    p_rms = cfg.DATA.P_RM
    for p_add in p_adds:
        for p_rm in p_rms:
            # Get dataset of dictionaries.
            # Each dictionary contains the informations
            # relative to one source-target graph pair.
            pair_dataset = SemiSyntheticDataset(root_dir=cfg.DATA.PATH,
                                                p_add=p_add,
                                                p_rm=p_rm,
                                                size=cfg.DATA.SIZE,
                                                train_ratio=cfg.DATA.TRAIN_RATIO)
            
            dataloader = DataLoader(pair_dataset)

            # --- Train & Evaluate ---
            matching_accuracies = []
            comp_times = []
            for i, pair_dict in enumerate(dataloader):
                start_time = time.time()
                
                alignment_matrix = align_networks(pair_dict, cfg, device=device)
                
                end_time = time.time()
                
                acc = compute_accuracy(alignment_matrix,
                                       pair_dict['test_dict'],
                                       matcher=cfg.MATCHER)
                
                print(f"Pair {i}, accuracy: ", acc.item())
                
                matching_accuracies.append(acc.item())
                comp_times.append(end_time - start_time)

            # Average computation time
            avg_time = np.mean(comp_times)

            # Average accuracy and std deviation
            avg_accuracy = np.mean(matching_accuracies)
            std_deviation = np.std(matching_accuracies)

            # Save output data
            output_data = [{'model': cfg.MODEL.NAME,
                           'data': data_name,
                           'p_add': p_add,
                           'p_rm': p_rm,
                           'num_tests': cfg.DATA.SIZE,
                           'avg_acc': avg_accuracy,
                           'std_dev': std_deviation,
                           'avg_time': avg_time}]

            print('Average matching accuracy: ', avg_accuracy)
            print('Std deviation: ', std_deviation)

            # Write result to CSV
            with open(csv_path, 'a', newline='') as outfile:
                csv_writer = csv.DictWriter(outfile, fieldnames=header)
                csv_writer.writerows(output_data)
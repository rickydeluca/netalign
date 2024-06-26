import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader

from netalign.data.dataset import RealDataset
from netalign.data.utils import dict_to_perm_mat, move_tensors_to_device
from netalign.evaluation.matchers import greedy_match
from netalign.evaluation.metrics import (compute_accuracy,
                                         compute_sim_prox_score)
from netalign.models import init_align_model


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description='Read command line arguments.')

    parser.add_argument('-s', '--source', type=str, required=True, help="Path to source network edgelist.")
    parser.add_argument('-t', '--target', type=str, required=True, help="Path to target network edgelist.")
    parser.add_argument('-c', '--cfg', type=str, required=True, help="Path to the model configuration (YAML file).")
    parser.add_argument('--gt', type=str, required=True, help="Path to the groundtruth file.")
    parser.add_argument('--num_exps', type=int, default=10, help="Number of experiments. Default: 10")
    parser.add_argument('--train_ratio', type=float, default=0.2, help="Percentage of nodes/graphs to use as training set. Default: 0.2")
    parser.add_argument('--val_ratio', type=float, default=0.0, help="Percentage of nodes/graphs to use as validation set. Default: 0.0")
    parser.add_argument('--seed', type=int, default=None, help="Seed for reproducibility. Default: None")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use. Default: cuda")
    parser.add_argument('--res_dir', type=str, default='results/real', help="Path to the directory where to save the test results. Default: results/real")
    parser.add_argument('--pred_dir', type=str, default='output/real', help="Path to the directory where to save the predicted alignments. Default: output/real")
    
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
    # Read input
    args = parse_args()
    cfg = read_config_file(args.cfg)

    # Set reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Load dataset
    dataset = RealDataset(
        source_path=args.source,
        target_path=args.target,
        gt_path=args.gt,
        seed=args.seed
    )

    dataloader = DataLoader(dataset, shuffle=False)

    # Init model
    model, model_name = init_align_model(cfg)

    # Init result file
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    
    source_net_name = args.source.split('/')[-1].split('.')[0]
    target_net_name = args.target.split('/')[-1].split('.')[0]

    res_file = f'{args.res_dir}/{model_name}_{source_net_name}_{target_net_name}.csv'

    header = ['model', 'source', 'target', 'num_experiments', 'avg_acc', 'std_acc', 'avg_sim_prox', 'std_sim_prox', 'avg_time']
    with open(res_file, 'w') as rf:
        csv_writer = csv.DictWriter(rf, fieldnames=header)
        csv_writer.writeheader()

    # Run experiments
    pair_dict = next(iter(dataloader))
    accs = []
    sim_proxs = []
    comp_times = []
    best_acc = 0
    best_align_matrix = None
    for exp_id in range(1, args.num_exps+1):
        # Move to device
        pair_dict = move_tensors_to_device(pair_dict, device)
        try:
            model = model.to(device)
        except: 
            pass

        # Align
        start_time = time.time()
        S, _ = model.align(pair_dict)
        elapsed_time = time.time() - start_time

        P = greedy_match(S)

        # Compute accuracy
        gt_test = dict_to_perm_mat(pair_dict['gt_test'], pair_dict['graph_pair'][0].num_nodes, pair_dict['graph_pair'][1].num_nodes).detach().cpu().numpy()
        acc = compute_accuracy(P, gt_test)

        # Compute similarity proximity
        sim_prox = compute_sim_prox_score(S, gt_test, P)

        print(f"\nExperiment: {exp_id}, Accuracy: {acc}, Similarity proximity: {sim_prox}")

        # Save best prediction
        if acc > best_acc:
            best_acc = acc
            best_align_matrix = P

        accs.append(acc)
        sim_proxs.append(sim_prox)
        comp_times.append(elapsed_time)

    # Average metrics
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)
    avg_sim_prox = np.mean(sim_proxs)
    std_sim_prox = np.std(sim_proxs)
    avg_time = np.mean(comp_times)

    print(f"\n\nMean accuracy: {avg_acc}, Standard deviation: {std_acc}")
    print(f"Mean similarity proximity: {avg_sim_prox}, Standard deviation: {std_sim_prox}")

    # Write results
    out_data = [{
        'model': model_name,
        'source': source_net_name,
        'target': target_net_name,
        'num_experiments': args.num_exps,
        'avg_acc': avg_acc,
        'std_acc': std_acc,
        'avg_sim_prox': avg_sim_prox,
        'std_sim_prox': std_sim_prox, 
        'avg_time': avg_time
    }]

    with open(res_file, 'a', newline='') as rf:
        csv_writer = csv.DictWriter(rf, fieldnames=header)
        csv_writer.writerows(out_data)

    # Save best predicted aligns
    if not os.path.exists(args.pred_dir):
        os.makedirs(args.pred_dir)

    nonzero_rows, nonzero_cols = best_align_matrix.nonzero()
    pred_file = f'{args.pred_dir}/{model_name}_{source_net_name}_{target_net_name}.txt'

    with open(pred_file, 'w') as pf:
        for src_idx, tgt_idx in zip(nonzero_rows, nonzero_cols):
            src_id = pair_dict['idx2id'][0][src_idx]
            tgt_id = pair_dict['idx2id'][1][tgt_idx]
            pf.write(f"{src_id} {tgt_id}\n")
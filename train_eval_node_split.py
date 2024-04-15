import argparse
import csv
import os
import time

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader

from netalign.data.dataset import RobustnessDataset, TopologyDataset
from netalign.data.utils import move_tensors_to_device, dict_to_perm_mat
from netalign.evaluation.matcher import greedy_match
from netalign.evaluation.metrics import compute_accuracy
from netalign.models import MAGNA, PALE, SHELLEY, SIGMA


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

    # Reproducibility
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Check output directory
    out_dir = f'results/{cfg.TEST}/{cfg.MODEL.NAME.lower()}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Choose between 'topology' and 'robustness' test
    if cfg.TEST.lower() == 'robustness':
        # Init output file
        if cfg.MODEL.NAME.lower() == 'shelley':
            outfile = os.path.join(out_dir, f'{cfg.INIT.FEATURES}-{cfg.EMBEDDING.MODEL}-{cfg.MATCHING.MODEL}_{cfg.DATA.NAME}_v{cfg.VERSION}.csv')
        else:
            outfile = os.path.join(out_dir, cfg.OUTFILE)

        header = ['model', 'data', 'p_add', 'p_rm', 'dataset_size', 'avg_acc', 'std_dev', 'avg_time', 'avg_best_epoch']
        with open(outfile, 'w') as of:
            csv_writer = csv.DictWriter(of, fieldnames=header)
            csv_writer.writeheader()

        # Run test
        p_adds = cfg.DATA.P_ADD
        p_rms = cfg.DATA.P_RM
        for p_add in p_adds:
            for p_rm in p_rms:
                
                # Get dataset and dataloader
                dataset = RobustnessDataset(
                    root_dir=cfg.DATA.PATH,
                    subset=cfg.DATA.SUBSET,
                    train_ratio=cfg.DATA.TRAIN_RATIO,
                    val_ratio=cfg.DATA.VAL_RATIO,
                    seed=cfg.SEED,
                    p_add=p_add,
                    p_rm=p_rm,
                    num_copies=cfg.DATA.NUM_COPIES,
                    gt_mode=cfg.DATA.GT_MODE,
                    permute=False if cfg.MODEL.NAME.lower() == 'sigma' else True
                )

                dataloader = DataLoader(dataset, shuffle=True)

                matching_accuracies = []
                comp_times = []
                best_epochs = []

                for i, pair_dict in enumerate(dataloader):

                    # Init alignment model
                    if cfg.MODEL.NAME.lower() == 'magna':
                        model = MAGNA(cfg)
                        model_name = f'magna_{cfg.MODEL.MEASURE.lower()}_p{cfg.MODEL.POPULATION_SIZE}_g{cfg.MODEL.NUM_GENERATIONS}'
                    elif cfg.MODEL.NAME.lower() == 'pale':
                        model = PALE(cfg)
                        model_name = f'pale_{cfg.MODEL.NAME.lower()}_{cfg.MODEL.MAPPING.MODE.lower()}{cfg.MODEL.MAPPING.NUM_HIDDEN}_{cfg.MODEL.MAPPING.LOSS_FUNCTION}'
                    elif cfg.MODEL.NAME.lower() == 'shelley':
                        model = SHELLEY(cfg)
                        model = model.to(device)
                        if cfg.MATCHING.MODEL == 'sgm':
                            model_name = f"shelley_{cfg.INIT.FEATURES}-{cfg.EMBEDDING.MODEL}-{cfg.MATCHING.MODEL}-beta{cfg.MATCHING.BETA}-tau{cfg.MATCHING.TAU}"
                        else:
                            model_name = f"shelley_{cfg.INIT.FEATURES}-{cfg.EMBEDDING.MODEL}-{cfg.MATCHING.MODEL}"
                    elif cfg.MODEL.NAME.lower() == 'sigma':
                        model = SIGMA(cfg)
                        model_name = 'sigma'
                    else:
                        raise ValueError(f'Invalid model: {cfg.MODEL.NAME.lower()}')
                    
                    # Move data to device
                    pair_dict = move_tensors_to_device(pair_dict, device)

                    # Align networks
                    start_time = time.time()

                    S, best_epoch = model.align(pair_dict)

                    elapsed_time = time.time() - start_time

                    # Add batch dimension if not present
                    if S.ndim == 2:
                        S = S[None, :]
                    
                    # Compute accuracy
                    batch_size = S.shape[0]
                    acc = 0.0

                    for i in range(batch_size):
                        pred_mat = greedy_match(S[i])
                        test_gt = pair_dict['test_gt'][i].detach().cpu().numpy()
                        acc += compute_accuracy(pred_mat, test_gt) / batch_size
                    
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
                    'model': model_name,
                    'data': cfg.DATA.NAME,
                    'p_add': p_add,
                    'p_rm': p_rm,
                    'dataset_size': len(dataset),
                    'avg_acc': avg_accuracy,
                    'std_dev': std_deviation,
                    'avg_time': avg_time,
                    'avg_best_epoch': avg_best_epoch,
                }]

                print('Average matching accuracy: ', avg_accuracy)
                print('Std deviation: ', std_deviation)

                # Write result to CSV
                with open(outfile, 'a', newline='') as of:
                    csv_writer = csv.DictWriter(of, fieldnames=header)
                    csv_writer.writerows(output_data)

    elif cfg.TEST.lower() == 'topology':
        dataset = TopologyDataset(
            root_dir=cfg.DATA.PATH,
            source_names=cfg.DATA.SOURCE_NAMES,
            target_names=cfg.DATA.TARGET_NAMES,
            train_ratio=cfg.DATA.TRAIN_RATIO,
            val_ratio=cfg.DATA.VAL_RATIO,
            num_copies=cfg.DATA.NUM_COPIES,
            seed=cfg.DATA.SEED
        )

        raise ValueError(f"Topology test not yet implemented!")
    
    else:
        raise ValueError(f'Invalid test: {cfg.EVALUATION.lower()}.')






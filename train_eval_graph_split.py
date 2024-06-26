import argparse
import csv
import time
import os

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from netalign.models import MAGNA, PALE
from netalign.data.dataset import RobustnessDataset, TopologyDataset
from netalign.data.dataloader import collate_fn


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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Init alignment model
    if cfg.MODEL.NAME.lower() == 'magna':
        model = MAGNA(cfg)
    elif cfg.MODEL.NAME.lower() == 'pale':
        model = PALE(cfg)
    elif cfg.MODEL.NAME.lower() == 'shelley':
        model = SHELLEY_G(cfg)
        model = model.to(device)
    else:
        raise ValueError(f'Invalid model: {cfg.MODEL.NAME.lower()}')
    
    # Output directory
    out_dir = f'results/{cfg.TEST}/graph_split/{cfg.MODEL.NAME.lower()}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Choose between 'topology' and 'robustness' test
    if cfg.TEST.lower() == 'robustness':
        # Init output file
        if cfg.MODEL.NAME.lower() == 'shelley':
            outfile = os.path.join(out_dir, f'{cfg.INIT.FEATURES}-{cfg.EMBEDDING.MODEL}-{cfg.MATCHING.MODEL}_{cfg.DATA.NAME}_v{cfg.VERSION}.csv')
        else:
            outfile = os.path.join(out_dir, cfg.OUTFILE)

        header = ['model', 'data', 'p_add', 'p_rm', 'train_size', 'val_size', 'test_size', 'best_val_acc', 'best_val_epoch', 'test_acc', 'training_time']
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
                    gt_mode=cfg.DATA.GT_MODE
                )

                # Train, val and test split
                train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])

                # Dataloaders
                train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn)
                test_loader = DataLoader(test_dataset, shuffle=False)

                # Train model
                start_time = time.time()
                best_val_acc, best_val_epoch = model.train_eval(train_loader, val_loader)
                training_time = time.time() - start_time
                
                # Test
                test_acc = model.evaluate(test_loader, use_acc=True)

                print(f"\nRemove noise: {p_rm}, Add noise: {p_add}, Accuracy: {test_acc}\n")

                # Save output data
                output_data = [{
                    'model': f'{cfg.INIT.FEATURES}-{cfg.EMBEDDING.MODEL}-{cfg.MATCHING.MODEL}',
                    'data': cfg.DATA.NAME,
                    'p_add': p_add,
                    'p_rm': p_rm,
                    'train_size': len(train_loader),
                    'val_size': len(val_loader),
                    'test_size': len(test_loader),
                    'best_val_acc': best_val_acc,
                    'best_val_epoch': best_val_epoch,
                    'test_acc': test_acc,
                    'training_time': training_time
                }]

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






import os

import numpy as np
import torch.nn as nn
from netalign.models.magna.utils import read_align_dict, pyg_to_edgelist


class MAGNA(nn.Module):
    def __init__(self, cfg):
        self.measure = cfg.MODEL.MEASURE
        self.population_size = cfg.MODEL.POPULATION_SIZE
        self.num_generations = cfg.MODEL.NUM_GENERATIONS
        self.outfile = 'netalign/mapping/magna/tmp/out'

    def forward(self, pair_dict):
        
        # Generate edge lists from pyg graphs.
        source_graph = pair_dict['graph_pair'][0]
        target_graph = pair_dict['graph_pair'][1]

        source_edgelist_path = 'netalign/mapping/magna/tmp/source.edgelist'
        target_edgelist_path = 'netalign/mapping/magna/tmp/target.edgelist'
        pyg_to_edgelist(source_graph, source_edgelist_path)
        pyg_to_edgelist(target_graph, target_edgelist_path)

        # In MAGNA the source network is the smaller graph.
        # Check if we need to switch source and target.
        reversed = False
        if source_graph.num_nodes <= target_graph.num_nodes:
            G_edgelist = source_edgelist_path
            H_edgelist = target_edgelist_path
        else:
            G_edgelist = target_edgelist_path
            H_edgelist = source_edgelist_path
            reversed = True

        # Run MAGNA++
        os.system(f"netalign/mapping/magna/app/magnapp_cli_linux64 -G {G_edgelist} -H {H_edgelist} -m {self.measure} -p {self.population_size} -n {self.num_generations} -o {self.outfile}")

        # Read output alignment.
        align_dict, weighted = read_align_dict(self.outfile, reverse=reversed)

        # Generate alignment matrix.
        align_mat = np.zeros((source_graph.num_nodes, target_graph.num_nodes))

        if weighted:
            for source, target_and_weight in align_dict.items():
                target, weight = target_and_weight
                align_mat[int(source), int(target)] = float(weight)
        
        else:
            for source, target in align_dict.items():
                align_mat[int(source), int(target)] = 1

        return align_mat
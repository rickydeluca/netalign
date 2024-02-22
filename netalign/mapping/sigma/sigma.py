import numpy as np
import torch
import torch.nn as nn

from netalign.mapping.sigma.gnns import GIN
from netalign.mapping.sigma.mapping_model import SigmaMapping


class SIGMA(nn.Module):
    def __init__(self, cfg):
        self.node_feature_dim = cfg.MODEL.EMBEDDING.NODE_FEATURE_DIM
        self.embedding_dim = cfg.MODEL.EMBEDDING.EMBEDDING_DIM

        self.T = cfg.MODEL.MAPPING.T
        self.miss_match_value = cfg.MODEL.MAPPING.MISS_MATCH_VALUE
        self.tau = cfg.MODEL.MAPPING.TAU
        self.n_sink_iters = cfg.MODEL.MAPPING.N_SINK_ITERS
        self.n_samples = cfg.MODEL.MAPPING.N_SAMPLES

        self.lr = cfg.TRAIN.LR
        self.l2norm = cfg.TRAIN.L2NORM
        self.epochs = cfg.TRAIN.EPOCHS

        self.device = torch.device(cfg.DEVICE)

    def align(self, pair_dict):
        # Construct model
        f_update = GIN(in_channels=self.node_feature_dim,
                       out_channels=self.embedding_dim,
                       dim=self.embedding_dim).to(self.device)
        
        self.model = SigmaMapping(f_update,
                             tau=self.tau,
                             n_sink_iter=self.n_sink_iters,
                             n_samples=self.n_samples).to(self.device)
        

        # Get costs and probs
        source_graph = pair_dict['graph_pair'][0]
        target_graph = pair_dict['graph_pair'][1]
        cost_s = source_graph.edge_index
        cost_s = source_graph.edge_index
        num_nodes = min([source_graph.num_nodes, target_graph.num_nodes])

        p_s, cost_s = self.pp(cost_s)
        p_t, cost_t = self.pp(cost_t)

        # Train & Evaluate
        self.train(p_s, cost_s, p_t, cost_t)
        
    
    def pp(self, cost_in):
        cost = np.array(cost_in.todense())
        p = cost.sum(-1, keepdims=True)
        p = p / p.sum()
        p = torch.FloatTensor(p).to(self.device)

        cost = np.where(cost != 0)
        cost = torch.LongTensor(cost).to(self.device)

        return p, cost
    

    def train(self, p_s, cost_s, p_t, cost_t):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2norm)

        for epoch in (range(self.epochs)):
            # forward model
            self.model.train()
            _, loss = self.model(p_s, cost_s, p_t, cost_t, self.T, miss_match_value=self.miss_match_value)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del loss

            # evaluate
            with torch.no_grad():
                self.model.eval()
                logits_t, _ =self.model(p_s, cost_s, p_t, cost_t, self.T, miss_match_value=self.miss_match_value)
                self.evaluate(logits_t, epoch)
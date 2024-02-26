import numpy as np
import torch
import torch.nn as nn

from netalign.mapping.sigma.function import predict
from netalign.mapping.sigma.gnns import GIN
from netalign.mapping.sigma.mapping_model import SigmaMapping

from torch_geometric.utils import to_dense_adj


class SIGMA(nn.Module):
    def __init__(self, cfg):
        super(SIGMA, self).__init__()
        # Network options
        self.node_feature_dim = cfg.MODEL.NODE_FEATURE_DIM
        self.embedding_dim = cfg.MODEL.EMBEDDING_DIM
        self.T = cfg.MODEL.T
        self.miss_match_value = cfg.MODEL.MISS_MATCH_VALUE
        # Gumbel Sinkhorn options
        self.tau = cfg.MODEL.TAU
        self.n_sink_iters = cfg.MODEL.N_SINK_ITERS
        self.n_samples = cfg.MODEL.N_SAMPLES
        # Training options
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
        cost_t = target_graph.edge_index
        self.num_nodes = min([source_graph.num_nodes, target_graph.num_nodes])

        p_s, cost_s = self.pp(cost_s)
        p_t, cost_t = self.pp(cost_t)

        # Train & Evaluate
        self.train(p_s, cost_s, p_t, cost_t)

        # Construct alignment matrix
        self.model.eval()
        with torch.no_grad():
            S, _ = self.model(p_s, cost_s, p_t, cost_t, self.T, miss_match_value=self.miss_match_value)
        
        self.S = S.detach().cpu().numpy()[:self.num_nodes, :self.num_nodes]
        return self.S
    
    def pp(self, cost_in):
        cost = to_dense_adj(cost_in).squeeze(0).numpy()
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


    def evaluate(self, log_alpha, epoch=0):
        matched_row, matched_col = predict(log_alpha, n=self.num_nodes, m=self.num_nodes)
        pair_names = []
        for i in range(matched_row.shape[0]):
            pair_names.append([matched_row[i], matched_col[i]])

        node_correctness = 0
        for pair in pair_names:
            if pair[0] == pair[1]:
                node_correctness += 1
        node_correctness /= self.num_nodes

        print('Epoch: %d, NC: %.1f' % (epoch+1, node_correctness * 100))
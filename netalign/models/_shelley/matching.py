import torch
import torch.nn as nn
import torch.nn.functional as F

from netalign.models.shelley.loss import ContrastiveLossWithAttention
from netalign.models.shelley.utils.lap_solvers import log_sinkhorn, gumbel_sinkhorn


class StableGM(nn.Module):
    def __init__(self, f_update, beta=0.1, n_sink_iters=10, tau=1.0):
        super(StableGM, self).__init__()
        self.f_update = f_update
        self.loss_fn = ContrastiveLossWithAttention()
        self.n_sink_iters = n_sink_iters
        self.beta = beta
        self.tau = tau

    def forward(self, graph_s, graph_t, gt_perm=None, train=False):
        # Generate embeddings
        h_s = self.f_update(graph_s.x, graph_s.edge_index, graph_s.edge_attr)
        h_t = self.f_update(graph_t.x, graph_t.edge_index, graph_t.edge_attr)

        if h_s.dim() == h_t.dim() == 2:
            h_s = h_s.unsqueeze(0)
            h_t = h_t.unsqueeze(0)

        # Cosine similarity
        h_s = h_s / h_s.norm(dim=-1, keepdim=True)
        h_t = h_t / h_t.norm(dim=-1, keepdim=True)
        sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))
        
        if train:
            # Sinkhorn ranking matrix
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)
            src_ns = torch.tensor(rank_mat.size(1)).unsqueeze(0)
            tgt_ns = torch.tensor(rank_mat.size(2)).unsqueeze(0)

            # Hardness attention loss
            gt_perm = torch.eye(len(batch_s), dtype=torch.float, device=sim_mat.device).unsqueeze(0)
            loss = self.loss_fn(rank_mat, gt_perm, src_ns, tgt_ns, self.beta)
            return loss
        else:
            return sim_mat
        
    def get_alignment(self):
        """
        Get alignment matrix between source and target graph.
        """
        h_s = self.f_update(self.graph_s.x, self.graph_s.edge_index, self.graph_s.edge_attr)
        h_t = self.f_update(self.graph_t.x, self.graph_t.edge_index, self.graph_t.edge_attr)

        h_s = h_s / h_s.norm(dim=-1, keepdim=True)
        h_t = h_t / h_t.norm(dim=-1, keepdim=True)
        sim_mat = torch.matmul(h_s, h_t.t())

        return sim_mat
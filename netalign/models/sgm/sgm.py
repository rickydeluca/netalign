import torch
import torch.nn as nn
import torch.nn.functional as F

import netalign.models.sgm.gumbel_sinkhorn_ops as gs
from netalign.models.sgm.stable_matching.loss_func import ContrastiveLossWithAttention
from netalign.models.sgm.stable_matching.sm_solvers.stable_marriage import stable_marriage


class SGM(nn.Module):
    def __init__(self, f_update, tau, n_sink_iter, n_samples):
        super(SGM, self).__init__()
        self.f_update = f_update
        self.tau = tau
        self.n_sink_iter = n_sink_iter
        self.n_samples = n_samples
        self.loss_fn = ContrastiveLossWithAttention()

    def forward(self, graph_s, graph_t, train=False, T=5, groundtruth=None, beta=0.1):
        # Read input graphs
        edge_index_s = graph_s.edge_index
        edge_index_t = graph_t.edge_index
        x_s = graph_s.x
        x_t = graph_t.x

        n_node_s = x_s.shape[0]
        n_node_t = x_t.shape[0]

        if train:
            loss = 0.0

            best_loss = float('inf')
            total_loss = 0 
            loss_count = 0

            sim_previous = None
            for _ in range(T):
                # Adjust feature importance
                if sim_previous is not None:
                    x_s_importance = sim_previous.sum(-1, keepdims=True)
                    x_t_importance = sim_previous.sum(-2, keepdims=True).transpose(-1, -2)
                else:
                    x_s_importance = torch.ones([n_node_s, 1]).to(x_s.device)
                    x_t_importance = torch.ones([n_node_t, 1]).to(x_s.device)

                # Refine features
                h_s = self.f_update(x_s, x_s_importance, edge_index_s)
                h_t = self.f_update(x_t, x_t_importance, edge_index_t)

                # Compute similarity matrix:
                # An alternative to the dot product similarity is the cosine similarity.
                # Scale 50.0 allows logits to be larger than the noise.
                h_s = h_s / h_s.norm(dim=-1, keepdim=True)
                h_t = h_t / h_t.norm(dim=-1, keepdim=True)
                log_alpha = h_s @ h_t.transpose(-1, -2) * 50.0
                # log_alpha_ = F.pad(log_alpha, (0, n_node_s, 0h_s @ h_t.transpose(-1, -2) * 50.0, n_node_t), value=0.0)

                # Compute Sinkhorn similarity matrix
                sim_new = gs.gumbel_sinkhorn(log_alpha, self.tau, self.n_sink_iter, self.n_samples, noise=True)[0, :n_node_s, :n_node_t]

                # Contrastive loss
                src_ns = torch.LongTensor([n_node_s])
                tgt_ns = torch.LongTensor([n_node_t])

                loss = self.loss_fn(sim_new.unsqueeze(0), groundtruth.unsqueeze(0), src_ns, tgt_ns, beta)
                total_loss += loss

                if loss < best_loss:
                    best_loss = loss
                    loss_count += 1
                    sim_previous = sim_new.mean(0).detach().clone()

            return total_loss / float(loss_count)

        else:
            x_s_importance = torch.ones([n_node_s, 1]).to(x_s.device)
            x_t_importance = torch.ones([n_node_t, 1]).to(x_s.device)

            # Propagate
            h_s = self.f_update(x_s, x_s_importance, edge_index_s)
            h_t = self.f_update(x_t, x_t_importance, edge_index_t)

            # Compute similarity matrix
            h_s = h_s / h_s.norm(dim=-1, keepdim=True)
            h_t = h_t / h_t.norm(dim=-1, keepdim=True)
            log_alpha = h_s @ h_t.transpose(-1, -2) * 50.0

            # Sinkhorn ranking matrix
            sim_mat = gs.gumbel_sinkhorn(log_alpha, self.tau, self.n_sink_iter, self.n_samples, noise=True)[0, :n_node_s, :n_node_t]

            # Prediction matrix
            src_ns = torch.LongTensor([n_node_s])
            tgt_ns = torch.LongTensor([n_node_t])
            pred_mat = stable_marriage(sim_mat, n1=src_ns, n2=tgt_ns)

            return pred_mat

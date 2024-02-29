import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch_geometric.utils import to_dense_adj
from netalign.models.shelley.mapping.gm_solvers import gumbel_sinkhorn


def reward_general(n1, n2, e1, e2, theta_in, miss_match_value=-1.0):
    e1_dense = to_dense_adj(e1, max_num_nodes=n1)
    e2_dense = to_dense_adj(e2, max_num_nodes=n2)

    theta = theta_in[:, :n1, :n2]

    r = e1_dense @ theta @ e2_dense.transpose(-1, -2) * theta
    r = r.sum([1, 2]).mean()

    if miss_match_value > 0.0:
        ones_1 = torch.ones([1, n1, 1]).to(theta_in.device)
        ones_2 = torch.ones([1, n2, 1]).to(theta_in.device)

        r_mis_1 = e1_dense @ ones_1 @ ones_2.transpose(-1, -2) * theta
        r_mis_1 = r_mis_1.sum([1, 2]).mean()

        r_mis_2 = ones_1 @ ones_2.transpose(-1, -2) @ e2_dense.transpose(-1, -2) * theta
        r_mis_2 = r_mis_2.sum([1, 2]).mean()

        r_mis = r_mis_1 + r_mis_2 - 2.0*r

        r_mis = r_mis * miss_match_value

        r = r - r_mis

    return r

def predict(log_alpha, n=None, m=None):
    log_alpha = log_alpha.cpu().detach().numpy()
    row, col = linear_sum_assignment(-log_alpha)

    if n is not None and m is not None:
        matched_mask = (row < n) & (col < m)
        row = row[matched_mask]
        col = col[matched_mask]

    return row, col

class SigmaLoss(nn.Module):
    def __init__(self, f_update, tau, n_sink_iter, n_samples):
        super(SigmaLoss, self).__init__()
        self.f_update = f_update
        self.tau = tau
        self.n_sink_iter = n_sink_iter
        self.n_samples = n_samples

    def forward(self, x_s, edge_index_s, x_t, edge_index_t, T, miss_match_value=-1.0):
        loss = 0.0
        loss_count = 0

        best_reward = float('-inf')
        best_logits = None

        n_node_s = x_s.shape[0]
        n_node_t = x_t.shape[0]

        theta_previous = None
        for _ in range(T):
            # Adjust feature importance
            if theta_previous is not None:
                x_s_importance = theta_previous.sum(-1, keepdims=True)
                x_t_importance = theta_previous.sum(-2, keepdims=True).transpose(-1, -2)
            else:
                x_s_importance = torch.ones([n_node_s, 1]).to(x_s.device)
                x_t_importance = torch.ones([n_node_t, 1]).to(x_s.device)

            # Propagate
            h_s = self.f_update(x_s, x_s_importance, edge_index_s)
            h_t = self.f_update(x_t, x_t_importance, edge_index_t)

            # An alternative to the dot product similarity is the cosine similarity.
            # scale 50.0 allows logits to be larger than the noise.
            h_s = h_s / h_s.norm(dim=-1, keepdim=True)
            h_t = h_t / h_t.norm(dim=-1, keepdim=True)
            log_alpha = h_s @ h_t.transpose(-1, -2) * 50.0

            log_alpha_ = F.pad(log_alpha, (0, n_node_s, 0, n_node_t), value=0.0)

            theta_new = gumbel_sinkhorn(log_alpha_, self.tau, self.n_sink_iter, self.n_samples, noise=True)

            r = reward_general(n_node_s, n_node_t, edge_index_s, edge_index_t, theta_new, miss_match_value=miss_match_value)

            if best_reward < r.item():
                best_reward = r.item()
                loss = loss - r
                loss_count += 1
                theta_previous = theta_new[:, :n_node_s, :n_node_t].mean(0).detach().clone()
                best_logits = log_alpha_.detach().clone()

        return best_logits, loss / float(loss_count)
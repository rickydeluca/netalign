import torch
import torch.nn as nn
import torch.nn.functional as F

from netalign.models.shelley.loss import ContrastiveLossWithAttention, EuclideanLoss
from netalign.models.shelley.utils.lap_solvers import log_sinkhorn, gumbel_sinkhorn
from netalign.models.shelley.utils.rewards import reward_general


def init_matching_module(f_update, cfg):
    if cfg.MODEL == 'sgm':
        model = StableGM(f_update=f_update,
                         beta=cfg.BETA,
                         n_sink_iters=cfg.N_SINK_ITERS,
                         tau=cfg.TAU,
                         mask=cfg.MASK)
        
    elif cfg.MODEL == 'sigma':
        model = SIGMA(f_update=f_update,
                      tau=cfg.TAU,
                      n_sink_iter=cfg.N_SINK_ITERS,
                      n_samples=cfg.N_SAMPLES,
                      T=cfg.T,
                      miss_match_value=cfg.MISS_MATCH_VALUE)
        
    elif cfg.MODEL == 'linear':
        model = LinearMapping(f_update=f_update,
                              embedding_dim=cfg.EMBEDDING_DIM)
        
    else:
        raise ValueError(f"Invalid matching model: {cfg.MATCHING.MODEL}")
    
    return model


class LinearMapping(nn.Module):
    def __init__(self, f_update, embedding_dim):
        super(LinearMapping, self).__init__()
        self.f_update = f_update
        self.loss_fn = EuclideanLoss()
        self.maps = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, graph_s, graph_t, train=False, train_dict=None):
        # Generate embeddings
        h_s = self.f_update(graph_s.x, graph_s.edge_index, graph_s.edge_attr)
        h_t = self.f_update(graph_t.x, graph_t.edge_index, graph_t.edge_attr)

        if h_s.dim() == h_t.dim() == 2:
            h_s = h_s.unsqueeze(0)
            h_t = h_t.unsqueeze(0)

        h_s = h_s / h_s.norm(dim=-1, keepdim=True)
        h_t = h_t / h_t.norm(dim=-1, keepdim=True)

        if train:
            # Map source to target
            batch_idx, source_idx, target_idx = torch.nonzero(train_dict['gt_perm'], as_tuple=True)
            source_feats = h_s[batch_idx, source_idx]
            target_feats = h_t[batch_idx, target_idx]

            mapped_source_feats = self.maps(source_feats)
            mapped_source_feats = F.normalize(mapped_source_feats, dim=-1)

            # Euclidean loss
            batch_size = source_feats.shape[0]
            loss = self.loss_fn(mapped_source_feats, target_feats) / batch_size

            # Similarity matrix
            sim_mat = torch.matmul(mapped_source_feats, target_feats.t())

            return sim_mat, loss
        
        else:
            h_s_map = F.normalize(self.maps(h_s), dim=-1)
            sim_mat = torch.matmul(h_s_map, h_t.transpose(1, 2))

            return sim_mat


class StableGM(nn.Module):
    def __init__(self, f_update, beta=0.1, n_sink_iters=10, tau=1.0, mask=False):
        super(StableGM, self).__init__()
        self.f_update = f_update
        self.loss_fn = ContrastiveLossWithAttention()
        self.n_sink_iters = n_sink_iters
        self.beta = beta
        self.tau = tau
        self.mask = mask

    def forward(self, graph_s, graph_t, train=False, train_dict=None):
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
            # Sinkhorn ranking on batch
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)

            # Hardness attention loss
            loss = self.loss_fn(rank_mat,
                                train_dict['gt_perm'], 
                                train_dict['src_ns'], 
                                train_dict['tgt_ns'],
                                self.beta,
                                mask=self.mask)
            
            return sim_mat, loss
        
        else:
            return sim_mat
        

class SIGMA(nn.Module):
    def __init__(self, f_update, tau, n_sink_iter, n_samples, T, miss_match_value):
        super(SIGMA, self).__init__()
        self.f_update = f_update
        self.tau = tau
        self.n_sink_iter = n_sink_iter
        self.n_samples = n_samples
        self.T = T
        self.miss_match_value = miss_match_value

    def forward(self, graph_s, graph_t, train=False, train_dict=None):
        loss = 0.0
        loss_count = 0

        best_reward = float('-inf')
        best_logits = None

        n_node_s = graph_s.x.shape[0]
        n_node_t = graph_t.x.shape[0]

        theta_previous = None
        for _ in range(self.T):
            # Adjust feature importance
            if theta_previous is not None:
                x_s_importance = theta_previous.sum(-1, keepdims=True)
                x_t_importance = theta_previous.sum(-2, keepdims=True).transpose(-1, -2)
            else:
                x_s_importance = torch.ones([n_node_s, 1]).to(graph_s.x.device)
                x_t_importance = torch.ones([n_node_t, 1]).to(graph_t.x.device)

            # Propagate
            h_s = self.f_update(graph_s.x, x_s_importance, graph_s.edge_index)
            h_t = self.f_update(graph_t.x, x_t_importance, graph_t.edge_index)

            # An alternative to the dot product similarity 
            # is the cosine similarity. Scale 50.0 allows
            # logits to be larger than the noise.
            h_s = h_s / h_s.norm(dim=-1, keepdim=True)
            h_t = h_t / h_t.norm(dim=-1, keepdim=True)
            log_alpha = h_s @ h_t.transpose(-1, -2) * 50.0

            log_alpha_ = F.pad(log_alpha, (0, n_node_s, 0, n_node_t), value=0.0)

            theta_new = gumbel_sinkhorn(log_alpha_, self.tau, self.n_sink_iter, self.n_samples, noise=True)

            r = reward_general(n_node_s, n_node_t,
                               graph_s.edge_index, graph_t.edge_index,
                               theta_new,
                               miss_match_value=self.miss_match_value)

            if best_reward < r.item():
                best_reward = r.item()
                loss = loss - r
                loss_count += 1
                theta_previous = theta_new[:, :n_node_s, :n_node_t].mean(0).detach().clone()
                best_logits = log_alpha_.detach().clone()

        if train:
            return best_logits, loss / float(loss_count)
        else:
            return best_logits

"""
class RandomStableGM(nn.Module):
    def __init__(self, f_update, beta=0.1, n_sink_iters=10, tau=1.0):
        super(RandomStableGM, self).__init__()
        self.f_update = f_update
        self.loss_fn = RandomContrastiveLossWithAttention()
        self.n_sink_iters = n_sink_iters
        self.beta = beta
        self.tau = tau

    @staticmethod
    def mask_indices(tensor, row_indices, col_indices):
        batch_size, N, M = tensor.size()
        mask = torch.zeros(batch_size, N, M, dtype=torch.bool, device=tensor.device)
        mask[:, row_indices, :] = True
        mask[:, :, col_indices] = True

        # Ensure only specified rows and columns are kept
        mask[:, :, [i for i in range(M) if i not in col_indices]] = False
        mask[:, [i for i in range(M) if i not in row_indices], :] = False
        return tensor.masked_fill(~mask, 1e-20)

    def forward(self, graph_s, graph_t, src_ns=None, tgt_ns=None, gt_perm=None, train=False):
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
        
        # Rescale similarity to remove negative values
        min_value = torch.min(sim_mat)
        sim_mat = sim_mat + torch.abs(min_value) + 1
        
        if train:
            # Mask and log similarity matrix
            _, row_indices, col_indices = torch.nonzero(gt_perm, as_tuple=True)

            sim_mat = self.mask_indices(sim_mat, row_indices, col_indices)
            sim_mat = torch.log(sim_mat)

            # Sinkhorn ranking matrix
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)

            # Hardness attention loss
            loss = self.loss_fn(rank_mat, gt_perm.to(torch.float), src_ns, tgt_ns, self.beta)
            return sim_mat, loss
        else:
            return sim_mat
        
class BatchStableGM(nn.Module):
    def __init__(self, f_update, beta=0.1, n_sink_iters=10, tau=1.0):
        super(BatchStableGM, self).__init__()
        self.f_update = f_update
        self.loss_fn = ContrastiveLossWithAttention()
        self.n_sink_iters = n_sink_iters
        self.beta = beta
        self.tau = tau

    def forward(self, graph_s, graph_t, batch_indices_s=None, batch_indices_t=None, train=False):
        # Generate embeddings
        h_s = self.f_update(graph_s.x, graph_s.edge_index, graph_s.edge_attr)[batch_indices_s]
        h_t = self.f_update(graph_t.x, graph_t.edge_index, graph_t.edge_attr)[batch_indices_t]

        if h_s.dim() == h_t.dim() == 2:
            h_s = h_s.unsqueeze(0)
            h_t = h_t.unsqueeze(0)

        # Cosine similarity
        h_s = h_s / h_s.norm(dim=-1, keepdim=True)
        h_t = h_t / h_t.norm(dim=-1, keepdim=True)
        sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

        # Sinkhorn ranking matrix
        rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)
        num_src_nodes = len(batch_indices_s) if batch_indices_s is not None else graph_s.num_nodes
        num_tgt_nodes = len(batch_indices_t) if batch_indices_t is not None else graph_t.num_nodes
        src_ns = torch.tensor(num_src_nodes).unsqueeze(0)
        tgt_ns = torch.tensor(num_tgt_nodes).unsqueeze(0)
        
        if train:
            # Hardness attention loss
            gt_perm = torch.eye(len(batch_indices_s), device=sim_mat.device).unsqueeze(0).to(torch.float)
            loss = self.loss_fn(rank_mat, gt_perm.to(torch.float), src_ns, tgt_ns, self.beta)
            return sim_mat, loss
        else:
            return sim_mat
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from netalign.models.shelley.loss import (
    ContrastiveLossWithAttention, EuclideanLoss, PermutationLoss,
    RandomContrastiveLossWithAttention)
from netalign.models.shelley.utils.lap_solvers import log_sinkhorn
from netalign.models.shelley.utils.sm_solvers.stable_marriage import \
    stable_marriage

def normalize_over_channels(x):
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms

class LinearMapping(nn.Module):
    def __init__(self, f_update, embedding_dim):
        super(LinearMapping, self).__init__()
        self.f_update = f_update
        self.loss_fn = EuclideanLoss()
        self.maps = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, graph_s, graph_t, src_ns=None, tgt_ns=None, gt_perm=None, train=False):
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
            batch_idx, source_idx, target_idx = torch.nonzero(gt_perm, as_tuple=True)
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
        
        
class CosineMapping(nn.Module):
    def __init__(self, f_update, n_sink_iters=10, tau=1.0):
        super(CosineMapping, self).__init__()
        self.f_update = f_update
        self.loss_fn = PermutationLoss()
        self.n_sink_iters = n_sink_iters
        self.tau = tau

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
        sim_mat = torch.matmul(h_s, h_t.transpose(1, 2)).clamp(0, 1)
        

        if train:
            rank_mat = log_sinkhorn(sim_mat, tau=self.tau, n_iter=self.n_sink_iters)
            loss = self.loss_fn(rank_mat, gt_perm, src_ns, tgt_ns)
            return sim_mat, loss
        
        else:
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

    def forward(self, graph_s, graph_t, src_ns=None, tgt_ns=None, train=False, gt_perm=None):
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
            loss = self.loss_fn(rank_mat, gt_perm.to(torch.float), src_ns, tgt_ns, self.beta, mask=self.mask)
            return sim_mat, loss
        else:
            return sim_mat


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
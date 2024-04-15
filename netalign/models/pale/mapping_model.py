import torch
import torch.nn as nn
import torch.nn.functional as F

from netalign.models.pale.loss import (ContrastiveLossWithAttention,
                                       MappingLossFunctions, PermutationLoss,
                                       TopContrastiveLossWithAttention)
from netalign.models.pale.utils.lap_solvers import log_sinkhorn


class PaleMappingMlp(nn.Module):
    """
    Class to handle both linear and mlp mapping models by specifying the
    number of hidden layers.
    """
    def __init__(self, embedding_dim, source_embedding, target_embedding, num_hidden_layers=0, activate_function='sigmoid', loss_function='contrastive', n_sink_iters=10, tau=1.0, beta=0.1, top_k=5):
        super(PaleMappingMlp, self).__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding

        if loss_function == 'contrastive':
            self.tau = tau
            self.beta = beta
            self.n_sink_iters = n_sink_iters
            self.loss_fn = ContrastiveLossWithAttention()
        elif loss_function == 'top_contrastive':
            self.tau = tau
            self.top_k = top_k
            self.n_sink_iters = n_sink_iters
            self.loss_fn = TopContrastiveLossWithAttention()
        elif loss_function == 'permutation':
            self.tau = tau
            self.n_sink_iters = n_sink_iters
            self.loss_fn = PermutationLoss()
        else:
            self.loss_fn = MappingLossFunctions()

        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers > 0:
            if activate_function == 'sigmoid':
                self.activate_function = nn.Sigmoid()
            elif activate_function == 'relu':
                self.activate_function = nn.ReLU()
            else:
                self.activate_function = nn.Tanh()

            hidden_dim = 2 * embedding_dim
            layers = [nn.Linear(embedding_dim, hidden_dim, bias=True), self.activate_function]

            for _ in range(num_hidden_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=True), self.activate_function])

            layers.append(nn.Linear(hidden_dim, embedding_dim, bias=True))
            self.mapping_network = nn.Sequential(*layers)
        else:
            self.mapping_network = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        source_feats_after_mapping = self.forward(source_feats)
        batch_size = source_feats.shape[0]

        if isinstance(self.loss_fn, ContrastiveLossWithAttention):
            # Similarity matrix
            h_s = source_feats_after_mapping / source_feats_after_mapping.norm(dim=-1, keepdim=True).unsqueeze(0)
            h_t = target_feats / target_feats.norm(dim=-1, keepdim=True).unsqueeze(0)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            # Ranking matrix
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)
            src_ns = torch.tensor(rank_mat.size(1)).unsqueeze(0)
            tgt_ns = torch.tensor(rank_mat.size(2)).unsqueeze(0)

            # Contrastive loss with attention
            gt_perm = torch.eye(batch_size, dtype=torch.float, device=sim_mat.device).unsqueeze(0)
            mapping_loss = self.loss_fn(rank_mat, gt_perm, src_ns, tgt_ns, self.beta)
        elif isinstance(self.loss_fn, TopContrastiveLossWithAttention):
            # Similarity matrix
            h_s = source_feats_after_mapping / source_feats_after_mapping.norm(dim=-1, keepdim=True).unsqueeze(0)
            h_t = target_feats / target_feats.norm(dim=-1, keepdim=True).unsqueeze(0)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            # Ranking matrix
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)
            src_ns = torch.tensor(rank_mat.size(1)).unsqueeze(0)
            tgt_ns = torch.tensor(rank_mat.size(2)).unsqueeze(0)

            # Contrastive loss with attention
            gt_perm = torch.eye(batch_size, dtype=torch.float, device=sim_mat.device).unsqueeze(0)
            mapping_loss = self.loss_fn(rank_mat, gt_perm, src_ns, tgt_ns, top_k=self.top_k)
        elif isinstance(self.loss_fn, PermutationLoss):
            # Similarity matrix
            h_s = source_feats_after_mapping / source_feats_after_mapping.norm(dim=-1, keepdim=True).unsqueeze(0)
            h_t = target_feats / target_feats.norm(dim=-1, keepdim=True).unsqueeze(0)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            # Ranking matrix
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)
            src_ns = torch.tensor(rank_mat.size(1)).unsqueeze(0)
            tgt_ns = torch.tensor(rank_mat.size(2)).unsqueeze(0)

            # Permutation loss
            gt_perm = torch.eye(batch_size, dtype=torch.float, device=sim_mat.device).unsqueeze(0)
            mapping_loss = self.loss_fn(rank_mat, gt_perm, src_ns, tgt_ns)
        else:
            # Euclidean Loss
            mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size

        return mapping_loss

    def forward(self, source_feats):
        if self.num_hidden_layers > 0:
            ret = self.mapping_network(source_feats)
        else:
            ret = self.mapping_network(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret
    

class PaleMappingMlp_Double(nn.Module):
    """
    Class to handle both linear and mlp mapping models by specifying the
    number of hidden layers.
    """
    def __init__(self, embedding_dim, source_embedding, target_embedding, num_hidden_layers=0, activate_function='sigmoid', loss_function='contrastive', n_sink_iters=10, tau=1.0, beta=0.1, top_k=5):
        super(PaleMappingMlp_Double, self).__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding

        if loss_function == 'contrastive':
            self.tau = tau
            self.beta = beta
            self.n_sink_iters = n_sink_iters
            self.loss_fn = ContrastiveLossWithAttention()
        elif loss_function == 'top_contrastive':
            self.tau = tau
            self.top_k = top_k
            self.n_sink_iters = n_sink_iters
            self.loss_fn = TopContrastiveLossWithAttention()
        elif loss_function == 'permutation':
            self.tau = tau
            self.n_sink_iters = n_sink_iters
            self.loss_fn = PermutationLoss()
        else:
            self.loss_fn = MappingLossFunctions()

        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers > 0:
            if activate_function == 'sigmoid':
                activate_function = nn.Sigmoid()
            elif activate_function == 'relu':
                activate_function = nn.ReLU()
            else:
                activate_function = nn.Tanh()

            hidden_dim = 2 * embedding_dim
            layers_st = [nn.Linear(embedding_dim, hidden_dim, bias=True), activate_function]
            layers_ts = [nn.Linear(embedding_dim, hidden_dim, bias=True), activate_function]

            for _ in range(num_hidden_layers - 1):
                layers_st.extend([nn.Linear(hidden_dim, hidden_dim, bias=True), activate_function])
                layers_ts.extend([nn.Linear(hidden_dim, hidden_dim, bias=True), activate_function])

            layers_st.append(nn.Linear(hidden_dim, embedding_dim, bias=True))
            layers_ts.append(nn.Linear(hidden_dim, embedding_dim, bias=True))

            self.map_st = nn.Sequential(*layers_st)
            self.map_ts = nn.Sequential(*layers_ts)
        else:
            self.map_st = nn.Linear(embedding_dim, embedding_dim, bias=True)
            self.map_ts = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def loss(self, source_indices, target_indices):
        source_feats = self.source_embedding[source_indices]
        target_feats = self.target_embedding[target_indices]

        h_s, h_t = self.forward(source_feats, target_feats)
        batch_size = source_feats.shape[0]

        if isinstance(self.loss_fn, ContrastiveLossWithAttention):
            # Similarity matrix
            h_s = h_s.unsqueeze(0)
            h_t = h_t.unsqueeze(0)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            # Ranking matrix
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)
            src_ns = torch.tensor(rank_mat.size(1)).unsqueeze(0)
            tgt_ns = torch.tensor(rank_mat.size(2)).unsqueeze(0)

            # Contrastive loss with attention
            gt_perm = torch.eye(batch_size, dtype=torch.float, device=sim_mat.device).unsqueeze(0)
            mapping_loss = self.loss_fn(rank_mat, gt_perm, src_ns, tgt_ns, self.beta)
        
        elif isinstance(self.loss_fn, TopContrastiveLossWithAttention):
            # Similarity matrix
            h_s = h_s.unsqueeze(0)
            h_t = h_t.unsqueeze(0)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            # Ranking matrix
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)
            src_ns = torch.tensor(rank_mat.size(1)).unsqueeze(0)
            tgt_ns = torch.tensor(rank_mat.size(2)).unsqueeze(0)

            # Contrastive loss with attention
            gt_perm = torch.eye(batch_size, dtype=torch.float, device=sim_mat.device).unsqueeze(0)
            mapping_loss = self.loss_fn(rank_mat, gt_perm, src_ns, tgt_ns, top_k=self.top_k)
        elif isinstance(self.loss_fn, PermutationLoss):
            # Similarity matrix
            h_s = h_s.unsqueeze(0)
            h_t = h_t.unsqueeze(0)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            # Ranking matrix
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)
            src_ns = torch.tensor(rank_mat.size(1)).unsqueeze(0)
            tgt_ns = torch.tensor(rank_mat.size(2)).unsqueeze(0)

            # Permutation loss
            gt_perm = torch.eye(batch_size, dtype=torch.float, device=sim_mat.device).unsqueeze(0)
            mapping_loss = self.loss_fn(rank_mat, gt_perm, src_ns, tgt_ns)
        else:
            # Euclidean Loss
            mapping_loss = self.loss_fn.loss(h_s, h_t) / batch_size

        return mapping_loss

    def forward(self, source_feats, target_feats):
        ret_st = self.map_st(source_feats)
        ret_ts = self.map_ts(target_feats)
        ret_st = F.normalize(ret_st, dim=1)
        ret_ts = F.normalize(ret_ts, dim=1)
        return ret_st, ret_ts
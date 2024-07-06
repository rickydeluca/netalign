import torch
import torch.nn as nn
import torch.nn.functional as F

from netalign.models.shelley.loss import ContrastiveLossWithAttention, PermutationLoss, MappingLossFunctions
from netalign.models.shelley.utils.lap_solvers import log_sinkhorn, gumbel_sinkhorn
from netalign.models.shelley.utils.sm_solvers.stable_marriage import stable_marriage
from netalign.models.shelley.utils.sm_solvers.greedy_match import greedy_match
from netalign.models.shelley.utils.rewards import reward_general
from torch_geometric.utils import to_dense_adj
from netalign.data.utils import get_valid_matrix_mask


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
    elif cfg.MODEL == 'stablesigma':
        model = StableSIGMA(f_update=f_update,
                      tau=cfg.TAU,
                      n_sink_iter=cfg.N_SINK_ITERS,
                      n_samples=cfg.N_SAMPLES,
                      T=cfg.T,
                      miss_match_value=cfg.MISS_MATCH_VALUE,
                      sup_loss='contrastive',
                      sup_loss_ratio=1.0)
    elif cfg.MODEL == 'palemap':
        model = PaleMappingMlp(
            f_update=f_update,
            embedding_dim=cfg.EMBEDDING_DIM,
            num_hidden_layers=cfg.NUM_LAYERS,
            activate_function=cfg.ACTIVATE_FUNCTION
        )
    elif cfg.MODEL == 'tpm':
        model = TPM(f_update=f_update)
        
    else:
        raise ValueError(f"Invalid matching model: {cfg.MATCHING.MODEL}")
    
    return model


class TPM(nn.Module):
    def __init__(self, f_update):
        super(TPM, self).__init__()
        self.f_update = f_update
        self.loss_fn = PermutationLoss()

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

        cross_sim = torch.matmul(h_s, h_t.transpose(1, 2))
        within_sim_1 = torch.matmul(h_s, h_s.transpose(1, 2))
        within_sim_2 = torch.matmul(h_t, h_t.transpose(1, 2))

        if train:
            # Triple Permutation Loss
            within_loss_1 = self.loss_fn(within_sim_1,
                                         to_dense_adj(graph_s.edge_index), 
                                         train_dict['src_ns'], 
                                         train_dict['src_ns'])/ graph_s.num_nodes
            
            within_loss_2 = self.loss_fn(within_sim_2,
                                         to_dense_adj(graph_t.edge_index), 
                                         train_dict['tgt_ns'], 
                                         train_dict['tgt_ns']) / graph_s.num_nodes
            
            cross_loss = self.loss_fn(cross_sim,
                                      train_dict['gt_perm'], 
                                      train_dict['src_ns'], 
                                      train_dict['tgt_ns']) / graph_s.num_nodes
            
            loss = within_loss_1 + within_loss_2 + cross_loss

            return cross_sim, loss
        
        else:
            return cross_sim


class _StableGM(nn.Module):
    def __init__(self, f_update, beta=0.1, n_sink_iters=10, tau=1.0, mask=False):
        super(_StableGM, self).__init__()
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
        

class StableSIGMA(nn.Module):
    def __init__(self, f_update, tau, n_sink_iter, n_samples, T, miss_match_value, sup_loss=None, sup_loss_ratio=0.4):
        super(StableSIGMA, self).__init__()
        self.f_update = f_update
        self.tau = tau
        self.n_sink_iter = n_sink_iter
        self.n_samples = n_samples
        self.T = T
        self.miss_match_value = miss_match_value
        self.sup_loss_fn = ContrastiveLossWithAttention() if sup_loss == 'contrastive' else None
        self.sup_loss_ratio = sup_loss_ratio

    def forward(self, graph_s, graph_t, train=False, train_dict=None):
        loss = 0.0
        loss_count = 0

        best_reward = float('-inf')
        best_logits = None

        n_node_s = graph_s.num_nodes
        n_node_t = graph_t.num_nodes
        
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
            h_s = self.f_update(graph_s.x, graph_s.edge_index, graph_s.edge_attr, x_importance=x_s_importance)
            h_t = self.f_update(graph_t.x, graph_t.edge_index, graph_t.edge_attr, x_importance=x_t_importance)

            # Scale 50.0 allows logits to be larger than the noise
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
                best_logits = log_alpha_

        # Get similarity matrix
        sim_mat = best_logits[:n_node_s, :n_node_t].unsqueeze(0)

        if train:
            qap_loss = loss/float(loss_count)
            
            if self.sup_loss_fn is not None:
                rank_mat = log_sinkhorn(sim_mat, n_iter=10, tau=1)
                sup_loss = self.sup_loss_fn(rank_mat,
                                            train_dict['gt_perm'],
                                            train_dict['src_ns'],
                                            train_dict['tgt_ns'],
                                            beta_value=0.1,
                                            mask=True)
                train_loss = qap_loss + self.sup_loss_ratio * sup_loss
            else:
                train_loss = qap_loss

            
            return sim_mat, train_loss
        
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

        n_node_s = graph_s.num_nodes
        n_node_t = graph_t.num_nodes

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
            h_s = self.f_update(graph_s.x, graph_s.edge_index, graph_s.edge_attr, x_importance=x_s_importance)
            h_t = self.f_update(graph_t.x, graph_t.edge_index, graph_t.edge_attr, x_importance=x_t_importance)

            # Scale 50.0 allows logits to be larger than the noise
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
            return best_logits[:n_node_s, :n_node_t], loss / float(loss_count)
        
        else:
            return best_logits[:n_node_s, :n_node_t]
        

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
 
        if train:
            # Take only the embeddings of training nodes
            valid_srcs, valid_tgts = train_dict['gt_perm'].squeeze(0).nonzero(as_tuple=True)
            h_s = h_s[valid_srcs]
            h_t = h_t[valid_tgts]

            # Generate corresponding groundtruth
            gt = torch.eye(valid_srcs.size(0)).unsqueeze(0).to(valid_srcs.device)
            src_ns = torch.tensor([valid_srcs.size(0)]).to(valid_srcs.device)
            tgt_ns = torch.tensor([valid_tgts.size(0)]).to(valid_tgts.device)

            if h_s.dim() == h_t.dim() == 2:
                h_s = h_s.unsqueeze(0)
                h_t = h_t.unsqueeze(0)

            # Cosine similarity
            h_s = h_s / h_s.norm(dim=-1, keepdim=True)
            h_t = h_t / h_t.norm(dim=-1, keepdim=True)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            # Sinkhorn ranking
            rank_mat = log_sinkhorn(sim_mat, n_iter=self.n_sink_iters, tau=self.tau)

            # Hardness attention loss
            loss = self.loss_fn(rank_mat,
                                gt, 
                                src_ns, 
                                tgt_ns,
                                self.beta,
                                mask=self.mask)
            
            return sim_mat, loss
        
        else:
            if h_s.dim() == h_t.dim() == 2:
                h_s = h_s.unsqueeze(0)
                h_t = h_t.unsqueeze(0)

            # Cosine similarity
            h_s = h_s / h_s.norm(dim=-1, keepdim=True)
            h_t = h_t / h_t.norm(dim=-1, keepdim=True)
            sim_mat = torch.matmul(h_s, h_t.transpose(1, 2))

            return sim_mat
        

class PaleMappingMlp(nn.Module):
    """
    Class to handle both linear and mlp mapping models by specifying the
    number of hidden layers.
    """
    def __init__(self, f_update=None, embedding_dim=256, num_hidden_layers=1, activate_function='sigmoid'):
        super(PaleMappingMlp, self).__init__()
        self.f_update = f_update
        self.source_embedding = None
        self.target_embedding = None
        self.loss_fn = MappingLossFunctions()

        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers > 0:   # MLP
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
        else:   # Linear
            self.mapping_network = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, graph_s, graph_t, train=False, train_dict=None):
        # GCN is optional in PaleMapping
        if self.f_update is not None:
            h_s = self.f_update(graph_s.x, graph_s.edge_index, graph_s.edge_attr)
            h_t = self.f_update(graph_t.x, graph_t.edge_index, graph_t.edge_attr)
        else:
            h_s = graph_s.x
            h_t = graph_s.x

        if train:
            # Map features
            source_indices, target_indices = train_dict['gt_perm'].squeeze(0).nonzero(as_tuple=True)
            source_feats = h_s[source_indices]
            target_feats = h_t[target_indices]
            source_feats_after_mapping = self.map(source_feats)

            # Compute euclidean loss
            batch_size = source_feats.shape[0]
            mapping_loss = self.loss_fn.loss(source_feats_after_mapping, target_feats) / batch_size

            return None, mapping_loss
        else:
            # Map features
            source_feats_after_mapping = self.map(h_s)
            target_feats = h_t

            if source_feats_after_mapping.dim() == target_feats.dim() == 2:
                source_feats_after_mapping = source_feats_after_mapping.unsqueeze(0)
                target_feats = target_feats.unsqueeze(0)

            # Cosine similarity
            source_feats_after_mapping = source_feats_after_mapping / source_feats_after_mapping.norm(dim=-1, keepdim=True)
            target_feats = target_feats / target_feats.norm(dim=-1, keepdim=True)
            sim_mat = torch.matmul(source_feats_after_mapping, target_feats.transpose(1, 2))
            return sim_mat
        

    def map(self, source_feats):
        if self.num_hidden_layers > 0:
            ret = self.mapping_network(source_feats)
        else:
            ret = self.mapping_network(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret
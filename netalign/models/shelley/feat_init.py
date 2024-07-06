import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import degree
from tqdm import tqdm

from netalign.models.pale.embedding_model import PaleEmbedding


def init_feat_module(cfg):
    # Feature initialization
    if cfg.TYPE.lower() == 'degree':
        f_init = Degree()
    elif cfg.TYPE.lower() == 'share':
        f_init = Share(cfg.FEATURE_DIM)
    elif cfg.TYPE.lower() == 'pale':
        f_init = Pale(cfg)
    else:
        raise ValueError(f"Invalid features: {cfg.TYPE}.")
    
    return f_init


class Degree(nn.Module):
    def __init__(self):
        super(Degree, self).__init__()

    def forward(self, graph):
        return degree(graph.edge_index[0], num_nodes=graph.num_nodes).unsqueeze(1)
    

class Share(nn.Module):
    def __init__(self, node_feature_dim):
        super(Share, self).__init__()
        self.node_feature_dim = node_feature_dim

    def forward(self, graph):
        return torch.ones((graph.num_nodes, self.node_feature_dim))
    

class Pale(nn.Module):
    def __init__(self, cfg):
        super(Pale, self).__init__()
        self.emb_batchsize = cfg.BATCH_SIZE
        self.emb_lr = cfg.LR
        self.neg_sample_size = cfg.NEG_SAMPLE_SIZE
        self.embedding_dim = cfg.EMBEDDING_DIM
        self.emb_epochs = cfg.EPOCHS
        self.embedding_name = cfg.EMBEDDING_NAME
        self.emb_optimizer = cfg.OPTIMIZER

    def forward(self, graph):
        # Init PALE
        num_nodes = graph.num_nodes
        edges = graph.edge_index.t().detach().cpu().numpy()
        deg = degree(graph.edge_index[0],
                     num_nodes=num_nodes).detach().cpu().numpy() 
        return self.learn_embedding(num_nodes, deg, edges)

    def learn_embedding(self, num_nodes, deg, edges):
        # Embedding model
        embedding_model = PaleEmbedding(n_nodes=num_nodes,
                                        embedding_dim=self.embedding_dim,
                                        deg=deg,
                                        neg_sample_size=self.neg_sample_size,
                                        cuda=self.cuda)
        if self.cuda:
            embedding_model = embedding_model.cuda()

        # Optimizer
        if self.emb_optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, embedding_model.parameters()),
                lr=self.emb_lr
            )
        else:
            raise ValueError(f'Invalid embedding optimizer: {self.emb_optimizer}.')
        
        # Train
        embedding = self.train_embedding(embedding_model, edges, optimizer)

        return embedding


    def train_embedding(self, embedding_model, edges, optimizer):
        n_iters = len(edges) // self.emb_batchsize
        assert n_iters > 0, "`batch_size` is too large."
        if(len(edges) % self.emb_batchsize > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        n_epochs = self.emb_epochs
        for epoch in tqdm(range(1, n_epochs + 1), desc="Init PALE feats:"):
            start = time.time()     # Time evaluation
            
            np.random.shuffle(edges)
            for iter in range(n_iters):
                batch_edges = torch.LongTensor(edges[iter*self.emb_batchsize:(iter+1)*self.emb_batchsize])
                if self.cuda:
                    batch_edges = batch_edges.cuda()
                start_time = time.time()
                optimizer.zero_grad()
                loss, loss0, loss1 = embedding_model.loss(batch_edges[:, 0], batch_edges[:,1])
                loss.backward()
                optimizer.step()
                total_steps += 1
            
            self.embedding_epoch_time = time.time() - start     # Time evaluation
            
        embedding = embedding_model.get_embedding()
        embedding = embedding.cpu().detach().numpy()
        embedding = torch.FloatTensor(embedding)
        if self.cuda:
            embedding = embedding.cuda()

        return embedding
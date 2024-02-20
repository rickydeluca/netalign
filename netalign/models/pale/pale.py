import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch_geometric

from netalign.models.pale.embedding_model import PaleEmbedding
from netalign.models.pale.mapping_model import (PaleMappingLinear,
                                                PaleMappingMlp)


class PALE(nn.Module):
    def __init__(self, cfg):
        self.emb_batchsize = cfg.MODEL.EMBEDDING.BATCH_SIZE
        self.map_batchsize = cfg.MODEL.MAPPING.BATCH_SIZE
        self.emb_lr = cfg.MODEL.EMBEDDING.LR
        self.cuda = True if 'cuda' in cfg.DEVICE else False
        self.neg_sample_size = cfg.MODEL.EMBEDDING.NEG_SAMPLE_SIZE
        self.embedding_dim = cfg.MODEL.EMBEDDING.EMBEDDING_DIM
        self.emb_epochs = cfg.MODEL.EMBEDDING.EPOCHS
        self.map_epochs = cfg.MODEL.MAPPING.EPOCHS
        self.mapping_model = cfg.MODEL.MAPPING.NAME
        self.map_act = cfg.MODEL.MAPPING.ACTIVATE_FUNCTION
        self.map_lr = cfg.MODEL.MAPPING.LR
        self.embedding_name = cfg.MODEL.EMBEDDING.EMBEDDING_NAME
        self.emb_optimizer = cfg.MODEL.EMBEDDING.OPTIMIZER
        self.map_optimizer = cfg.MODEL.MAPPING.OPTIMIZER


    def align(self, pair_dict):
        # Learn embeddings
        self.source_graph = pair_dict['graph_pair'][0]
        self.target_graph = pair_dict['graph_pair'][1]
        self.learn_embeddings()

        # Define mapping model
        if self.mapping_model == 'linear':
            print("Use linear mapping")
            mapping_model = PaleMappingLinear(embedding_dim=self.embedding_dim,
                                              source_embedding=self.source_embedding,
                                              target_embedding=self.target_embedding)
        elif self.mapping_model == 'mlp':
            print("Use Mpl mapping")
            mapping_model = PaleMappingMlp(embedding_dim=self.embedding_dim,
                                           source_embedding=self.source_embedding,
                                           target_embedding=self.target_embedding,
                                           activate_function=self.map_act)
        else:
            raise ValueError(f'Invalid mapping model: {self.mapping_model}.')
        
        if self.cuda:
            mapping_model = mapping_model.cuda()

        # Mapping optimizer
        if self.map_optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, mapping_model.parameters()),
                                         lr=self.map_lr)
        else:
            raise ValueError(f'Invalid mapping optmizer: {self.map_optimizer}.')
        
        # --- Configure training ---
        # Groundtruth
        self.gt_train = pair_dict['train_dict']
        self.source_train_nodes = np.array(list(self.gt_train.keys()))

        # Mini-batching
        n_iters = len(self.source_train_nodes) // self.map_batchsize
        assert n_iters > 0, "`batch_size` is too large."
        if(len(self.source_train_nodes) % self.map_batchsize > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        n_epochs = self.map_epochs

        # --- Map training --- 
        for epoch in range(1, n_epochs + 1):
            start = time.time()     # Time evaluation

            print('Epochs: ', epoch)
            np.random.shuffle(self.source_train_nodes)
            for iter in range(n_iters):
                source_batch = self.source_train_nodes[iter*self.map_batchsize:(iter+1)*self.map_batchsize]
                target_batch = [self.gt_train[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch)
                target_batch = torch.LongTensor(target_batch)
                if self.cuda:
                    source_batch = source_batch.cuda()
                    target_batch = target_batch.cuda()
                optimizer.zero_grad()
                start_time = time.time()
                loss = mapping_model.loss(source_batch, target_batch)
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0 and total_steps > 0:
                    print("Iter:", '%03d' %iter,
                          "train_loss=", "{:.5f}".format(loss.item()),
                          "time", "{:.5f}".format(time.time()-start_time))
                total_steps += 1

            self.mapping_epoch_time = time.time() - start   # Time evaluation

        self.source_after_mapping = mapping_model(self.source_embedding)
        self.S = torch.matmul(self.source_after_mapping, self.target_embedding.t())
        
        self.S = self.S.detach().cpu().numpy()

        return self.S


    def get_source_embedding(self):
        return self.source_embedding


    def get_target_embedding(self):
        return self.target_embedding


    def learn_embeddings(self):
        # Init embedding learning
        num_source_nodes = self.source_graph.num_nodes
        source_edges = self.source_graph.edge_index.t().numpy() 
        source_deg = torch_geometric.utils.degree(self.source_graph.edge_index[0],
                                                  num_nodes=num_source_nodes).numpy()

        num_target_nodes = self.target_graph.num_nodes
        target_edges = self.target_graph.edge_index.t().numpy() 
        target_deg = torch_geometric.utils.degree(self.target_graph.edge_index[0],
                                                  num_nodes=num_target_nodes).numpy()
        
        self.source_embedding = self.learn_embedding(num_source_nodes, source_deg, source_edges)
        self.target_embedding = self.learn_embedding(num_target_nodes, target_deg, target_edges)


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
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                embedding_model.parameters()),
                                        lr=self.emb_lr)
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
        for epoch in range(1, n_epochs + 1):
            start = time.time()     # Time evaluation

            print("Epoch {0}".format(epoch))
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
                if total_steps % print_every == 0:
                    print("Iter:", '%03d' %iter,
                          "train_loss=", "{:.5f}".format(loss.item()),
                          "true_loss=", "{:.5f}".format(loss0.item()),
                          "neg_loss=", "{:.5f}".format(loss1.item()),
                          "time", "{:.5f}".format(time.time()-start_time))
                total_steps += 1
            
            self.embedding_epoch_time = time.time() - start     # Time evaluation
            
        embedding = embedding_model.get_embedding()
        embedding = embedding.cpu().detach().numpy()
        embedding = torch.FloatTensor(embedding)
        if self.cuda:
            embedding = embedding.cuda()

        return embedding
import torch
import torch.nn as nn
import torch.optim as optim

from netalign.mapping.tau.gine import GINE
from netalign.mapping.tau.loss import MappingLossFunctions


class Tau(nn.Module):
    def __init__(self, cfg):
        super(Tau, self).__init__()

        # Embedding model
        self.gine = GINE(in_channels=cfg.GINE.NODE_FEATURE_DIM,
                         out_channels=cfg.GINE.EMBEDDING_DIM,
                         dim=cfg.GINE.EMBEDDING_DIM,
                         num_layers=cfg.GINE.NUM_LAYERS,
                         bias=cfg.GINE.BIAS)
        
        self.node_feature_dim = cfg.GINE.NODE_FEATURE_DIM
        
        # Training
        self.lr = cfg.TRAIN.LR
        self.l2norm = cfg.TRAIN.L2NORM
        self.epochs = cfg.TRAIN.EPOCHS
        self.loss_fn = MappingLossFunctions()

    def align(self, pair_dict):
        self.source_graph = pair_dict['graph_pair'][0]
        self.target_graph = pair_dict['graph_pair'][1]
        self.source_train_nodes = torch.LongTensor(list(pair_dict['train_dict'].keys()))
        self.target_train_nodes = torch.LongTensor(list(pair_dict['train_dict'].values()))

        # Init node features
        self.source_graph.x = torch.ones(
            (self.source_graph.num_nodes,
             self.node_feature_dim)
        )
        
        self.target_graph.x = torch.ones(
            (self.source_graph.num_nodes,
             self.node_feature_dim)
        )
        
        # Learn embeddings
        self.learn_embeddings()

        # Align
        self.gine.eval()
        with torch.no_grad():
            source_embeddings = self.gine(X=self.source_graph.x,
                                          edge_index=self.source_graph.edge_index,
                                          edge_attr=self.source_graph.edge_attr)
            
            target_embeddings = self.gine(X=self.source_graph.x,
                                          edge_index=self.source_graph.edge_index,
                                          edge_attr=self.source_graph.edge_attr)
            
            self.S = torch.matmul(source_embeddings, target_embeddings.t())
            self.S = self.S.detach().cpu().numpy()

        return self.S


    def learn_embeddings(self):
        optimizer = optim.Adam(self.gine.parameters(), lr=self.lr, weight_decay=self.l2norm)

        for epoch in range(1, self.epochs + 1):
            # Forward
            self.gine.train()
            source_embeddings = self.gine(X=self.source_graph.x,
                                          edge_index=self.source_graph.edge_index,
                                          edge_attr=self.source_graph.edge_attr)
            
            target_embeddings = self.gine(X=self.source_graph.x,
                                          edge_index=self.source_graph.edge_index,
                                          edge_attr=self.source_graph.edge_attr)
            
            # Loss
            source_train_embeddings = source_embeddings[self.source_train_nodes]
            target_train_embeddings = target_embeddings[self.target_train_nodes]
            loss = self.loss_fn.loss(source_train_embeddings, target_train_embeddings) / source_train_embeddings.shape[0]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(f'Epoch: {epoch}, Loss: {loss}')
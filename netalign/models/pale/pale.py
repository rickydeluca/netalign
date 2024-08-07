import time

import numpy as np
import torch
import torch.nn as nn
import torch_geometric
from tqdm import tqdm

from netalign.models.pale.embedding_model import PaleEmbedding
from netalign.models.pale.mapping_model import PaleMappingMlp


class PALE(nn.Module):
    def __init__(self, cfg):
        super(PALE, self).__init__()
        # Device
        self.cuda = True if 'cuda' in cfg.DEVICE else False

        # Embedding
        self.emb_batchsize = cfg.EMBEDDING.BATCH_SIZE
        self.emb_lr = cfg.EMBEDDING.LR
        self.embedding_dim = cfg.EMBEDDING.EMBEDDING_DIM
        self.emb_epochs = cfg.EMBEDDING.EPOCHS
        self.embedding_name = cfg.EMBEDDING.EMBEDDING_NAME
        self.emb_optimizer = cfg.EMBEDDING.OPTIMIZER
        self.neg_sample_size = cfg.EMBEDDING.NEG_SAMPLE_SIZE

        # Mapping
        self.map_train_batchsize = cfg.MAPPING.BATCH_SIZE_TRAIN
        self.map_val_batchsize = cfg.MAPPING.BATCH_SIZE_VAL
        self.map_validate = cfg.MAPPING.VALIDATE
        self.map_patience = cfg.MAPPING.PATIENCE        
        self.map_epochs = cfg.MAPPING.EPOCHS
        self.map_hidden_layers = cfg.MAPPING.NUM_HIDDEN
        self.map_act = cfg.MAPPING.ACTIVATE_FUNCTION
        self.map_lr = cfg.MAPPING.LR
        self.map_optimizer = cfg.MAPPING.OPTIMIZER
        self.map_l2norm = cfg.MAPPING.L2NORM
        self.map_loss_fn = cfg.MAPPING.LOSS_FUNCTION
        self.map_tau = cfg.MAPPING.TAU
        self.map_beta = cfg.MAPPING.BETA
        self.map_top_k = cfg.MAPPING.TOP_K

        self.S = None
        self.source_embedding = None
        self.target_embedding = None

    def align(self, pair_dict, verbose=False):
        # Representation phase
        self.verbose = verbose
        self.encode(pair_dict)

        # Mapping phase
        self.map(pair_dict)

        # Get final alignment
        self.get_alignment()
        
        return self.S, self.best_epoch
    
    def pprint(self, str):
        if self.verbose:
            print(str)
    
    def get_source_embedding(self):
        return self.source_embedding

    def get_target_embedding(self):
        return self.target_embedding
    
    def get_alignment(self):
        self.mapping_model.eval()
        self.source_after_mapping = self.mapping_model(self.source_embedding)
        self.S = torch.matmul(self.source_after_mapping, self.target_embedding.t())
        self.S = self.S.detach().cpu().numpy()
    
    def encode(self, pair_dict):
        """
        Call this function to generate the node representation
        without the mapping step.
        """
        self.source_graph = pair_dict['graph_pair'][0]
        self.target_graph = pair_dict['graph_pair'][1]
        self.learn_embeddings()

    def map(self, pair_dict):
        """
        Map the source and target node representations
        """

        # Check if node representations where computed
        if self.source_embedding is None or self.target_embedding is None:
            raise RuntimeError("Call `encode()` first.")
        
        # Define mapping model
        self.mapping_model = PaleMappingMlp(
            embedding_dim=self.embedding_dim,
            source_embedding=self.source_embedding,
            target_embedding=self.target_embedding,
            num_hidden_layers=self.map_hidden_layers,
            activate_function=self.map_act,
            loss_function=self.map_loss_fn,
            beta=self.map_beta,
            top_k=self.map_top_k
        )

        # Move to device
        if self.cuda:
            self.mapping_model = self.mapping_model.cuda()

        # Define optimizer
        if self.map_optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.mapping_model.parameters()),
                                        lr=self.map_lr)
        else:
            raise ValueError(f'Invalid mapping optimizer: {self.map_optimizer}.')

        # --- CONFIGURE MINI-BATCHING ---

        # Train
        self.gt_train = pair_dict['gt_train']
        self.source_train_nodes = np.array(list(self.gt_train.keys()))

        n_train_iters = len(self.source_train_nodes) // self.map_train_batchsize
        if n_train_iters == 0:
            n_train_iters = len(self.source_train_nodes)
            
        if len(self.source_train_nodes) % self.map_train_batchsize > 0:
            n_train_iters += 1
        print_every = int(n_train_iters / 4) + 1
        total_steps = 0

        # Validation
        if self.map_validate:
            self.gt_val = pair_dict['gt_val']
            self.source_val_nodes = np.array(list(self.gt_val.keys()))

            n_val_iters = len(self.source_val_nodes) // self.map_val_batchsize
            assert n_val_iters > 0, "`map_val_batchsize` is too large."
            if len(self.source_val_nodes) % self.map_val_batchsize > 0:
                n_val_iters += 1

            # Early stopping
            best_val_loss = float('inf')
            self.best_epoch = -1
            best_state_dict = {}
            patience = self.map_patience


        # --- TRAIN & EVAL LOOP ---


        for epoch in range(1, self.map_epochs + 1):
            start = time.time()  # Time evaluation
            self.pprint(f'Epoch: {epoch}')

            # Train
            self.mapping_model.train()
            np.random.shuffle(self.source_train_nodes)
            for iter in range(n_train_iters):
                source_train_batch = self.source_train_nodes[iter * self.map_train_batchsize:(iter + 1) * self.map_train_batchsize]
                target_train_batch = [self.gt_train[x] for x in source_train_batch]
                source_train_batch = torch.LongTensor(source_train_batch)
                target_train_batch = torch.LongTensor(target_train_batch)
                
                if self.cuda:
                    source_train_batch = source_train_batch.cuda()
                    target_train_batch = target_train_batch.cuda()

                optimizer.zero_grad()
                start_time = time.time()
                loss = self.mapping_model.loss(source_train_batch, target_train_batch)
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0 and total_steps > 0 and self.verbose:
                    print("Iter:", '%03d' % iter, 
                          "train_loss:", "{:.5f}".format(loss.item()), 
                          "time:", "{:.5f}".format(time.time() - start_time))
                total_steps += 1

            # Validation
            if self.map_validate:
                val_loss = 0.0
                self.mapping_model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    for iter in range(n_val_iters):
                        source_val_batch = self.source_val_nodes[iter * self.map_val_batchsize:(iter + 1) * self.map_val_batchsize]
                        target_val_batch = [self.gt_val[x] for x in source_val_batch]
                        source_val_batch = torch.LongTensor(source_val_batch)
                        target_val_batch = torch.LongTensor(target_val_batch)
                        
                        if self.cuda:
                            source_val_batch = source_val_batch.cuda()
                            target_val_batch = target_val_batch.cuda()

                        val_batch_loss = self.mapping_model.loss(source_val_batch, target_val_batch).item()
                        val_loss += val_batch_loss / n_val_iters

                if self.verbose:
                    print("Validation Loss: {:.5f}".format(val_loss))

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    best_state_dict = self.mapping_model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.pprint(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
                    break

            self.mapping_epoch_time = time.time() - start  # Time evaluation

        if self.map_validate:
            # Load best model state dict
            self.mapping_model.load_state_dict(best_state_dict)
        else:
            self.best_epoch = -1

    def learn_embeddings(self):
        # Init embedding learning
        num_source_nodes = self.source_graph.num_nodes
        source_edges = self.source_graph.edge_index.t().detach().cpu().numpy() 
        source_deg = torch_geometric.utils.degree(self.source_graph.edge_index[0],
                                                  num_nodes=num_source_nodes).detach().cpu().numpy() 

        num_target_nodes = self.target_graph.num_nodes
        target_edges = self.target_graph.edge_index.t().detach().cpu().numpy() 
        target_deg = torch_geometric.utils.degree(self.target_graph.edge_index[0],
                                                  num_nodes=num_target_nodes).detach().cpu().numpy() 
        
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
        for epoch in tqdm(range(1, n_epochs + 1), desc='Embedding generation:'):
            start = time.time()     # Time evaluation

            if self.verbose:
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
                if total_steps % print_every == 0 and self.verbose:
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
    
    def get_embeddings(self):
        if self.source_embedding is not None and self.target_embedding is not None:
            return self.source_after_mapping.detach(), self.target_embedding
        else:
            raise ValueError("Call `encode()` method first!")
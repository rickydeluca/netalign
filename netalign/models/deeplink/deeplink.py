import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from netalign.data.utils import replace_tensor_items
from netalign.models.deeplink.embedding_model import DeepWalk
from netalign.models.deeplink.mapping_model import MappingModel


class DeepLink(nn.Module):
    def __init__(self, cfg):
        super(DeepLink, self).__init__()
        self.cuda = cfg.CUDA
        self.num_cores = cfg.NUM_CORES
        
        self.embedding_dim = cfg.EMBEDDING.EMBEDDING_DIM
        self.number_walks = cfg.EMBEDDING.NUMBER_WALKS
        self.walk_length = cfg.EMBEDDING.WALK_LENGTH
        self.window_size = cfg.EMBEDDING.WINDOW_SIZE
        self.embedding_epochs = cfg.EMBEDDING.EPOCHS

        self.alpha = cfg.MAPPING.ALPHA
        self.map_batchsize = cfg.MAPPING.BATCH_SIZE
        self.supervised_epochs = cfg.MAPPING.SUPERVISED_EPOCHS
        self.unsupervised_epochs = cfg.MAPPING.UNSUPERVISED_EPOCHS
        self.supervised_lr = cfg.MAPPING.SUPERVISED_LR
        self.unsupervised_lr = cfg.MAPPING.UNSUPERVISED_LR
        self.top_k = cfg.MAPPING.TOP_K
        self.hidden_dim1 = cfg.MAPPING.HIDDEN_DIM1
        self.hidden_dim2 = cfg.MAPPING.HIDDEN_DIM2

        self.S = None
        self.source_embedding = None
        self.target_embedding = None
        self.source_after_mapping = None

    def get_alignment_matrix(self):
        return self.S

    def get_source_embedding(self):
        return self.source_embedding

    def get_target_embedding(self):
        return self.target_embedding

    def align(self, pair_dict, verbose=False):
        # Read input
        self.verbose = verbose
        self.source_graph = pair_dict['graph_pair'][0]
        self.target_graph = pair_dict['graph_pair'][1]
        self.source_id2idx = replace_tensor_items(pair_dict['id2idx'][0])
        self.target_id2idx = replace_tensor_items(pair_dict['id2idx'][1])

        self.train_dict = pair_dict['gt_train']
        val_dict = pair_dict['gt_val']
        test_dict = pair_dict['gt_test']
        
        self.full_gt = {}
        self.full_gt.update(self.train_dict)
        self.full_gt.update(val_dict)
        self.full_gt.update(test_dict)

        self.source_train_nodes = np.array(list(self.train_dict.keys()))
        self.source_anchor_nodes = np.array(list(self.train_dict.keys()))

        # Embedding generation
        self.learn_embeddings()

        # Embedding mapping
        mapping_model = MappingModel(
            embedding_dim=self.embedding_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            source_embedding=self.source_embedding,
            target_embedding=self.target_embedding
        )

        if self.cuda:
            mapping_model = mapping_model.cuda()

        m_optimizer_us = torch.optim.SGD(filter(lambda p: p.requires_grad, mapping_model.parameters()), lr = self.unsupervised_lr)
        
        self.mapping_train_(mapping_model, m_optimizer_us, 'us')

        m_optimizer_s = torch.optim.SGD(filter(lambda p: p.requires_grad, mapping_model.parameters()), lr = self.supervised_lr)
        self.mapping_train_(mapping_model, m_optimizer_s, 's')
        self.source_after_mapping = mapping_model(self.source_embedding, 'val')
        self.S = torch.matmul(self.source_after_mapping, self.target_embedding.t())

        self.S = self.S.detach().cpu().numpy()
        return self.S, -1


    def mapping_train_(self, model, optimizer, mode='s'):
        if mode == 's':
            source_train_nodes = self.source_train_nodes
        else:
            source_train_nodes = self.source_anchor_nodes

        batch_size = self.map_batchsize
        n_iters = len(source_train_nodes)//batch_size
        assert n_iters > 0, "batch_size is too large"
        if(len(source_train_nodes) % batch_size > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        train_dict = None
        if mode == 's':
            n_epochs = self.supervised_epochs
            train_dict = self.train_dict
        else:
            n_epochs = self.unsupervised_epochs
            train_dict = self.full_gt


        for epoch in tqdm(range(1, n_epochs+1), desc='Mapping'):
            start = time.time()     # Time evaluation

            if self.verbose:
                print("Epoch {0}".format(epoch))
            np.random.shuffle(source_train_nodes)
            for iter in range(n_iters):
                source_batch = source_train_nodes[iter*batch_size:(iter+1)*batch_size]
                target_batch = [train_dict[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch)
                target_batch = torch.LongTensor(target_batch)
                if self.cuda:
                    source_batch = source_batch.cuda()
                    target_batch = target_batch.cuda()
                optimizer.zero_grad()
                start_time = time.time()
                if mode == 'us':
                    loss = model.unsupervised_loss(source_batch, target_batch)
                else:
                    loss = model.supervised_loss(source_batch, target_batch, alpha=self.alpha, k=self.top_k)
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0 and total_steps > 0 and self.verbose:
                    print("Iter:", '%03d' %iter,
                          "train_loss=", "{:.5f}".format(loss.item()),
                          )
            
                total_steps += 1
            if mode == "s":
                self.s_mapping_epoch_time = time.time() - start
            else:
                self.un_mapping_epoch_time = time.time() - start

    def learn_embeddings(self):
        print("Start embedding for source nodes, using deepwalk")

        # For evaluating time
        start = time.time()

        source_embedding_model = DeepWalk(
            self.source_graph, self.source_id2idx,
            self.number_walks, self.walk_length,
            self.window_size, self.embedding_dim,
            self.num_cores, self.embedding_epochs
        )
        
        self.source_embedding = torch.Tensor(source_embedding_model.get_embedding())
        
        
        self.embedding_epoch_time = time.time() - start

        print("Start embedding for target nodes, using deepwalk")

        target_embedding_model = DeepWalk(
            self.target_graph, self.target_id2idx,
            self.number_walks, self.walk_length,
            self.window_size, self.embedding_dim,
            self.num_cores, self.embedding_epochs
        )

        self.target_embedding = torch.Tensor(target_embedding_model.get_embedding())

        if self.cuda:
            self.source_embedding = self.source_embedding.cuda()
            self.target_embedding = self.target_embedding.cuda()

    @torch.no_grad()
    def get_embeddings(self):
        return self.source_after_mapping.detach(), self.target_embedding

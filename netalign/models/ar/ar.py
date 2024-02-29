import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import negative_sampling, degree
from torch_geometric.transforms import RandomLinkSplit

from netalign.models.ar.gnns import GCN, GINE
from netalign.models.ar.mapping import LinearMapper, MlpMapper


class AR(nn.Module):
    def __init__(self, cfg):
        super(AR, self).__init__()
        # Model settings
        self.model_name = cfg.MODEL.NAME
        self.node_feature_init = cfg.MODEL.NODE_FEATURE_INIT
        self.node_feature_dim = cfg.MODEL.NODE_FEATURE_DIM
        self.embedding_dim = cfg.MODEL.EMBEDDING_DIM
        self.hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.num_conv_layers = cfg.MODEL.NUM_LAYERS
        
        # Train settings
        self.emb_lr = cfg.TRAIN.EMBEDDING.LR
        self.emb_l2norm = cfg.TRAIN.EMBEDDING.L2NORM
        self.emb_epochs = cfg.TRAIN.EMBEDDING.EPOCHS

        self.map_lr = cfg.TRAIN.MAPPING.LR
        self.map_l2norm = cfg.TRAIN.MAPPING.L2NORM
        self.map_epochs = cfg.TRAIN.MAPPING.EPOCHS

        # Default device
        self.device = torch.device(cfg.DEVICE)

    def align(self, pair_dict):
        # Learn embeddings
        self.learn_embeddings(pair_dict)
        
        # Learn mapping
        self.learn_mapping(pair_dict)
        
        return self.S


    def learn_embeddings(self, pair_dict):
        # Get graphs
        self.source_graph = pair_dict['graph_pair'][0]
        self.target_graph = pair_dict['graph_pair'][1]

        # Init node features
        if self.node_feature_init == 'share':
            self.source_graph.x = torch.ones((self.source_graph.num_nodes, self.node_feature_dim))
            self.target_graph.x = torch.ones((self.target_graph.num_nodes, self.node_feature_dim))
        elif self.node_feature_init == 'degree':
            self.source_graph.x = degree(self.source_graph.edge_index[0],
                                        num_nodes=self.source_graph.num_nodes).unsqueeze(1)
            self.target_graph.x = degree(self.target_graph.edge_index[0],
                                        num_nodes=self.target_graph.num_nodes).unsqueeze(1)

        # Learn embeddings
        self.source_embeddings = self.train_embedding_model(self.source_graph)
        self.target_embeddings = self.train_embedding_model(self.target_graph)


    def train_embedding_model(self, graph):
        print('graph:', graph)
        # Get embedding model
        if 'gcn' in self.model_name:
            embedding_model = GCN(
                in_channels=self.node_feature_dim,
                hidden_channels=self.hidden_dim,
                out_channels=self.embedding_dim,
                num_layers=self.num_conv_layers
            )
        elif 'gine' in self.model_name:
            embedding_model = GINE(
                in_channels=self.node_feature_dim,
                dim=self.embedding_dim,
                out_channels=self.embedding_dim,
                num_conv_layers=self.num_conv_layers
            )
        else:
            raise ValueError(f'Invalid model: {self.model_name}')
        
        # Define optimizer
        optimizer = optim.Adam(embedding_model.parameters(), lr=self.emb_lr, weight_decay=self.emb_l2norm)

        # Loss function
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # Generate dataset by splitting the links and
        # adding the negative samples
        splitter = RandomLinkSplit(num_val=0.4, num_test=0.0, is_undirected=True,
                                   add_negative_train_samples=True,
                                   neg_sampling_ratio=2.0)
        
        train, val, _ = splitter(graph)

        # Shuffle indices and labels
        shuff_train = torch.randperm(train.edge_label_index.shape[1])
        train_indices = train.edge_label_index[:, shuff_train]
        train_labels = train.edge_label[shuff_train]

        shuff_val = torch.randperm(val.edge_label_index.shape[1])
        val_indices = val.edge_label_index[:, shuff_val]
        val_labels = val.edge_label[shuff_val]

        # Setup for early stopping
        best_val_loss = float('inf')
        best_state_dict = {}
        patience = 10
        patience_count = 0

        for epoch in range(self.emb_epochs):
            # --- Train ---
            embedding_model.train()

            # Forward step on the whole graph
            h_train = embedding_model(graph.x, graph.edge_index, graph.edge_attr)

            # Compute loss only on training indices
            h_src_train = h_train[train_indices[0]]
            h_dst_train = h_train[train_indices[1]]
            train_pred = (h_src_train * h_dst_train).sum(dim=-1)  # Inner product

            train_loss = criterion(train_pred, train_labels)

            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # --- Evaluate ---
            embedding_model.eval()

            with torch.no_grad():
                h_val = embedding_model(graph.x, graph.edge_index, graph.edge_attr)
                
                h_src_val = h_val[val_indices[0]]
                h_dst_val = h_val[val_indices[1]]
                val_pred = (h_src_val * h_dst_val).sum(dim=-1)  # Inner product

                val_loss = criterion(val_pred, val_labels)


            print(f'Epoch: {epoch+1}/{self.emb_epochs}, Train loss: {train_loss}, Val loss: {val_loss}')

            
            # Update best loss and params
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = embedding_model.state_dict()
            else:
                patience_count += 1
            
            # Check early stopping
            if patience_count >= patience:
                print('Early stop!')
                break

        # Load best model parameters
        embedding_model.load_state_dict(best_state_dict)

        # Get embeddings
        return self.get_embeddings(embedding_model, graph)


    @torch.no_grad()
    def get_embeddings(self, model, graph):
        model.eval()
        h = model(graph.x, graph.edge_index, graph.edge_attr)
        return h



    def learn_mapping(self, pair_dict):
        # Define mapping model
        if 'linear' in self.model_name:
            mapping_model = LinearMapper(embedding_dim=self.embedding_dim,
                                         source_embedding=self.source_embeddings,
                                         target_embedding=self.target_embeddings)
        elif 'mlp' in self.model_name:
            mapping_model = MlpMapper(embedding_dim=self.embedding_dim,
                                      source_embedding=self.source_embeddings,
                                      target_embedding=self.target_embeddings)
        else:
            raise ValueError(f'Invalid model: {self.model_name}.')
        
        # Define optimizer
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, mapping_model.parameters()),
                                     lr=self.map_lr)
        
        # Get groundtruth alignments
        gt_aligns = torch.tensor([list(pair_dict['train_dict'].keys()),
                                  list(pair_dict['train_dict'].values())], dtype=torch.long)
        shuffle = torch.randperm(gt_aligns.shape[1])
        gt_aligns = gt_aligns[:, shuffle]

        for epoch in range(self.map_epochs):
            mapping_model.train()

            # Forward & Loss
            optimizer.zero_grad()
            loss = mapping_model.loss(gt_aligns[0], gt_aligns[1])

            # Backward
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch+1}/{self.map_epochs}, Loss: {loss}')

        
        # Get final alignment matrix
        mapping_model.eval()
        with torch.no_grad():
            mapped_source_embeddings = mapping_model(self.source_embeddings)

        self.S = torch.matmul(mapped_source_embeddings, self.target_embeddings.t())
        self.S = self.S.detach().cpu().numpy()
    



"""
    def train_eval(self, source_graph, target_graph, pos_aligns):
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Sample negative alignments
        neg_aligns = negative_sampling(pos_aligns, force_undirected=True)
        all_aligns = torch.cat((pos_aligns, neg_aligns), dim=1)

        # Define groundtruth labels
        num_true_aligns = pos_aligns.shape[1]
        num_neg_aligns = neg_aligns.shape[1]
        gt_labels = torch.cat((torch.ones(num_true_aligns), torch.zeros(num_neg_aligns)))

        # Shuffle alignments and labels
        shuff_indices = torch.randperm(all_aligns.shape[1])
        all_aligns = all_aligns[:, shuff_indices]
        gt_labels = gt_labels[shuff_indices]

        for epoch in range(self.epochs):
            self.model.train()

            # Forward
            hs = self.model(source_graph.x, source_graph.edge_index, source_graph.edge_attr)
            ht = self.model(target_graph.x, target_graph.edge_index, target_graph.edge_attr)

            # Extract training subsets
            hs_train = hs[all_aligns[0]]
            ht_train = ht[all_aligns[1]]

            # Predict alignment
            pred_aligns = (hs_train * ht_train).sum(dim=-1)
            
            # Loss
            loss = F.binary_cross_entropy_with_logits(pred_aligns, gt_labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print or log the loss for monitoring
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")


        return self.model
    
    @torch.no_grad()
    def get_alignment(self, source_graph, target_graph):
        self.model.eval()
        hs = self.model(source_graph.x, source_graph.edge_index, source_graph.edge_attr)
        ht = self.model(target_graph.x, target_graph.edge_index, target_graph.edge_attr)
        S = torch.matmul(hs, ht.t()).detach().cpu().numpy()
        return S
"""
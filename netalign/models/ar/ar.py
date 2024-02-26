import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import negative_sampling

from netalign.models.ar.gnns import GCN, GIN


class AR(nn.Module):
    def __init__(self, cfg):
        super(AR, self).__init__()
        # GNN settings
        self.in_channels = cfg.MODEL.IN_CHANNELS
        self.hidden_channels = cfg.MODEL.HIDDEN_CHANNELS
        self.out_channels = cfg.MODEL.OUT_CHANNELS
        self.num_layers = cfg.MODEL.NUM_LAYERS
        # Train settings
        self.lr = cfg.TRAIN.LR
        self.l2norm = cfg.TRAIN.L2NORM
        self.epochs = cfg.TRAIN.EPOCHS
        self.device = torch.device(cfg.DEVICE)

    def align(self, pair_dict):
        # Construct model
        self.model = GCN(in_channels=self.in_channels, hidden_channels=self.hidden_channels,
                         out_channels=self.out_channels, num_layers=self.num_layers)
        # self.model = GIN(in_channels=self.in_channels, out_channels=self.out_channels, dim=self.out_channels)
        
        # Get graphs
        source_graph = pair_dict['graph_pair'][0]
        target_graph = pair_dict['graph_pair'][1]
        train_aligns = torch.tensor([list(pair_dict['train_dict'].keys()),
                                     list(pair_dict['train_dict'].values())], dtype=torch.long)
        
        # Init node features
        source_graph.x = torch.ones((source_graph.num_nodes, self.in_channels))
        target_graph.x = torch.ones((target_graph.num_nodes, self.in_channels))

        # Train model
        self.train_eval(source_graph, target_graph, train_aligns)

        # Get alignment matrix
        self.S = self.get_alignment(source_graph, target_graph)
        
        return self.S

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



        


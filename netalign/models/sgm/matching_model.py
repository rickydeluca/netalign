import torch
import torch.nn as nn
from torch_geometric.utils import degree

from netalign.models.sgm.gnns import GIN
from netalign.models.sgm.sgm import SGM


class SgmMatcher(nn.Module):
    '''
    Base class to handle matching models.    
    '''
    
    def __init__(self, cfg):
        super(SgmMatcher, self).__init__()

        # Configuration dictionary
        self.cfg = cfg

        # Default device
        self.device = torch.device(cfg.DEVICE)

        # Alignment attributes
        self.source_embeddings = None
        self.target_embeddings = None
        self.S = None

    def align(self, pair_dict):
        # Init graph features
        source_graph = self.to_device(pair_dict['graph_pair'][0])
        target_graph = self.to_device(pair_dict['graph_pair'][1])

        source_graph.x = self.init_features(source_graph)
        target_graph.x = self.init_features(target_graph)

        # Construct matching model
        embedding_model = GIN(
            in_channels=self.cfg.NETWORK.NODE_FEATURE_DIM,
            out_channels=self.cfg.NETWORK.DIM,
            dim=self.cfg.NETWORK.DIM
        ).to(self.device)

        self.matching_model = SGM(
            f_update=embedding_model,
            tau=self.cfg.SINKHORN.TAU,
            n_sink_iter=self.cfg.SINKHORN.N_SINK_ITERS,
            n_samples=self.cfg.SINKHORN.N_SAMPLES
        ).to(self.device)
        
        # Get train groundtruth alignment matrix
        gt_train = self.dict2mat(pair_dict['train_dict'],
                                 source_graph.num_nodes,
                                 target_graph.num_nodes)
        
        # Train model
        self.train_eval(source_graph, target_graph, gt_train)
        
        # Get alignment matrix
        self.S = self.get_alignment(source_graph, target_graph)

        return self.S
    
    def to_device(self, graph):
        """
        Move the PyTorch Data structure to the default device.
        """
        graph.x = graph.x.to(self.device) if graph.x is not None else None
        graph.edge_index = graph.edge_index.to(self.device)
        graph.edge_attr = graph.edge_attr.to(self.device)
        graph.batch = graph.batch.to(self.device)
        graph.ptr = graph.ptr.to(self.device)
        return graph
    
    @staticmethod
    def dict2mat(dictionary, n_rows, n_cols):
        matrix = torch.zeros((n_rows, n_cols))

        for s, t in dictionary.items():
            matrix[s, t] = 1
        
        return matrix
    
    def init_features(self, graph):
        return degree(graph.edge_index[0], num_nodes=graph.num_nodes).unsqueeze(1)
    
    def train_eval(self, source_graph, target_graph, gt_train):
        # Define optimizer
        optimizer = torch.optim.Adam(
            self.matching_model.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.L2NORM)
        
        for epoch in range(self.cfg.TRAIN.EPOCHS):
            # Forward step
            self.matching_model.train()
            loss = self.matching_model(
                graph_s=source_graph,
                graph_t=target_graph,
                train=True,
                T=self.cfg.NETWORK.T,
                groundtruth=gt_train,
                beta=self.cfg.NETWORK.BETA
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch}/{self.cfg.TRAIN.EPOCHS}, Loss: {loss}')


    
    @torch.no_grad()
    def get_alignment(self, source_graph, target_graph):
        self.matching_model.eval()
        pred_mat = self.matching_model(source_graph, target_graph)
        return pred_mat
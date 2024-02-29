import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.utils import degree
from torch_geometric.transforms import RandomLinkSplit

from netalign.models.shelley.mapping.gnns import GCN, GINE
from netalign.models.shelley.mapping.loss import BCEWithLogitsLoss, BCELoss
from netalign.models.shelley.embedding import ShareEmbedding, DegreeEmbedding, PaleEmbedding
from netalign.models.shelley.training.train_embedding import train_pale
from netalign.models.shelley.training.train_mapping import train_gnn


class SHELLEY(nn.Module):
    def __init__(self, cfg):
        super(SHELLEY, self).__init__()
        # Configuration dictionary
        self.cfg = cfg

        # Default device
        self.device = torch.device(cfg.DEVICE)

        # Alignment attributes
        self.source_embeddings = None
        self.target_embeddings = None
        self.S = None

    def align(self, pair_dict):
        # Get graphs
        source_graph = self.to_device(pair_dict['graph_pair'][0])
        target_graph = self.to_device(pair_dict['graph_pair'][1])

        # Learn embeddings
        self.source_embeddings = self.learn_embeddings(source_graph)
        self.target_embeddings = self.learn_embeddings(target_graph)

        # Update graph features
        source_graph.x = self.source_embeddings
        target_graph.x = self.target_embeddings

        # Learn mapping
        gt_aligns = torch.tensor([list(pair_dict['train_dict'].keys()),
                                  list(pair_dict['train_dict'].values())], dtype=torch.long)
        
        self.S = self.learn_mapping(source_graph, target_graph, gt_aligns)
        
        return self.S
    
    def to_device(self, graph):
        graph.x = graph.x.to(self.device) if graph.x is not None else None
        graph.edge_index = graph.edge_index.to(self.device)
        graph.edge_attr = graph.edge_attr.to(self.device)
    
    def learn_embeddings(self, graph):
        # Define embedding model
        if self.cfg.EMBEDDING.MODEL == 'pale':
            deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes).numpy()
            cuda = True if 'cuda' in self.cfg.DEVICE else False
            embedding_model = PaleEmbedding(
                n_nodes=graph.num_nodes,
                embedding_dim=self.cfg.EMBEDDING.EMBEDDING_DIM,
                deg=deg,
                neg_sample_size=self.cfg.EMBEDDING.NEG_SAMPLE_SIZE
            ).to(self.device)
        elif self.cfg.EMBEDDING.MODEL == 'share':
            embedding_model = ShareEmbedding(
                node_feature_dim=self.cfg.EMBEDDIN.NODE_FEATURE_DIM
            ).to(self.device)
        elif self.cfg.EMBEDDING.MODEL == 'degree':
            embedding_model = DegreeEmbedding(
                node_feature_dim=self.cfg.EMBEDDING.NODE_FEATURE_DIM
            ).to(self.device)
        else:
            raise ValueError(F"Invalid embedding model: {self.cfg.EMBEDDING.MODEL}.")
        
        # Define optimizer
        if self.cfg.EMBEDDING.OPTIMIZER.lower() == 'adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, embedding_model.parameters()),
                lr=self.cfg.EMBEDDING.LR,
                weight_decay=self.cfg.EMBEDDING.L2NORM
            )
        elif self.cfg.EMBEDDING.OPTIMIZER is None:
            optimizer = None
        else:
            raise ValueError(f"Invalid embedding optimizer: {self.cfg.EMBEDDING.OPTIMIZER.lower()}.")
        
        # Train embedding model
        embedding_model = self.train_embedding_model(model=embedding_model,
                                                     optimizer=optimizer,
                                                     graph=graph)
        
        # Get learned embeddings
        embeddings = self.get_embeddings(embedding_model, graph)

        return embeddings

    
    def train_embedding_model(self, model, optimizer, graph):
        """
        Train the embedding model `model` if it is trainable
        and return it, otherwise simply return the input model.
        """
        # Train embedding model
        if self.cfg.EMBEDDING.MODEL == 'pale':
            edges = graph.edge_index.t().numpy()
            model = train_pale(
                model=model,
                optimizer=optimizer,
                edges=edges,
                epochs=self.cfg.EMBEDDING.EPOCHS,
                batch_size=self.cfg.EMBEDDING.BATCH_SIZE
            )
        elif self.cfg.EMBEDDING.MODEL == 'share':
            pass
        elif self.cfg.EMBEDDING.MODEL == 'degree':
            pass
        else:
            raise ValueError(f"Invalid embedding model: {self.cfg.EMBEDDING.MODEL}.")
        
        return model
    
    @torch.no_grad()
    def get_embeddings(self, model, graph):
        """
        Generate the node embeddings for the input `graph`
        using the embedding `model`.
        """
        model.eval()

        if self.cfg.EMBEDDING.MODEL == 'pale':
            embeddings = model.get_embedding()
            embeddings = embeddings.cpu().detach().numpy()
            embeddings = torch.FloatTensor(embeddings).to(self.device)
        elif self.cfg.EMBEDDING.MODEL == 'share':
            embeddings = model(graph)
        elif self.cfg.EMBEDDING.MODEL == 'degree':
            embeddings = model(graph)
        else:
            raise ValueError(f"Invalid embedding model: {self.cfg.EMBEDDING.MODEL}.")

        return embeddings
    

    def learn_mapping(self, source_graph, target_graph, gt_aligns):
        # Define mapping model
        if self.cfg.MAPPING.MODEL == 'gcn':
            mapping_model = GCN(
                in_channels=self.cfg.MAPPING.IN_CHANNELS,
                hidden_channels=self.cfg.MAPPING.HIDDEN_CHANNELS,
                out_channels=self.cfg.MAPPING.OUT_CHANNELS,
                num_layers=self.cfg.MAPPING.NUM_LAYERS,
                normalize=self.cfg.MAPPING.NORMALIZE,
                bias=self.cfg.MAPPING.BIAS
            ).to(self.device)
        elif self.cfg.MAPPING.MODEL == 'gine':
            mapping_model = GINE(
                in_channels=self.cfg.MAPPING.IN_CHANNELS,
                dim=self.cfg.MAPPING.DIM,
                out_channels=self.cfg.MAPPING.OUT_CHANNELS,
                num_conv_layers=self.cfg.MAPPING.NUM_CONV_LAYERS,
                bias=self.cfg.MAPPING.BIAS
            ).to(self.device)
        else:
            raise ValueError(f"Invalid mapping model: {self.cfg.MAPPING.MODEL}.")
        
        # Define optimizer
        if self.cfg.MAPPING.OPTIMIZER.lower() == 'adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, mapping_model.parameters()),
                lr=self.cfg.EMBEDDING.LR,
                weight_decay=self.cfg.EMBEDDING.L2NORM
            )
        elif self.cfg.EMBEDDING.OPTIMIZER is None:
            optimizer = None
        else:
            raise ValueError(f"Invalid mapping optimizer: {self.cfg.MAPPING.OPTIMIZER.lower()}.")
        
        # Define loss function
        if self.cfg.MAPPING.LOSS_FN.lower() == 'bcewithlogits':
            loss_fn = BCEWithLogitsLoss()
        elif self.cfg.MAPPING.LOSS_FN.lower() == 'bce':
            loss_fn = BCELoss()
        else:
            print(f"Invalid loss function: {self.cfg.MAPPING.LOSS_FN.lower()}.")
        
        # Train mapping model
        mapping_model = self.train_mapping_model(
            model=mapping_model,
            optimizer=optimizer,
            source_graph=source_graph,
            target_graph=target_graph,
            gt_aligns=gt_aligns,
            loss_fn=loss_fn
        )

        # Get alignment matrix
        S = self.get_alignment(mapping_model, source_graph, target_graph)
        
        return S

    def train_mapping_model(self, model, optimizer, source_graph, target_graph, gt_aligns, loss_fn):
        """
        Train the model model `model` if it is trainable
        and return it, otherwise simply return the input model.
        """

        if self.cfg.MAPPING.MODEL == 'gcn':
            model = train_gnn(
                model=model,
                optimizer=optimizer,
                source_graph=source_graph,
                target_graph=target_graph,
                gt_aligns=gt_aligns,
                epochs=self.cfg.MAPPING.EPOCHS,
                loss_fn=loss_fn
            )
        elif self.cfg.MAPPING.MODEL == 'gine':
            model = train_gnn(
                model=model,
                optimizer=optimizer,
                source_graph=source_graph,
                target_graph=target_graph,
                gt_aligns=gt_aligns,
                epochs=self.cfg.MAPPING.EPOCHS,
                loss_fn=loss_fn
            )
        else:
            raise ValueError(f"Invalid mapping model: {self.cfg.EMBEDDING.MODEL}.")
        
        return model

    @torch.no_grad()
    def get_alignment(self, model, source_graph, target_graph):
        model.eval()
        hs = model(source_graph.x, source_graph.edge_index, source_graph.edge_attr)
        ht = model(target_graph.x, target_graph.edge_index, target_graph.edge_attr)
        S = torch.matmul(hs, ht.t()).detach().cpu().numpy()
        return S
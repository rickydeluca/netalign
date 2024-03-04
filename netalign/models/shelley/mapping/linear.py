import torch.nn as nn

class BaseMapping(nn.Module):
    """
    Class for a base mapping model that refine the input features
    using a GNN and align them using the groundtruth aligmnets.
    """
    def __init__(self, gnn, source_graph, target_graph, gt_aligns, loss_fn):
        super(BaseMapping, self).__init__()
        self.gnn = gnn
        self.source_graph = source_graph
        self.target_graph = target_graph
        self.gt_aligns = gt_aligns
        self.loss_fn = loss_fn

    def forward(self, source_batch, target_batch):
        # Refine source and target embeddings
        h_src = self.gnn(x=self.source_graph.x,
                         edge_index=self.source_graph.edge_index,
                         edge_attr=self.source_graph.edge_attr)
        h_tgt = self.gnn(x=self.target_graph.x,
                         edge_index=self.target_graph.edge_index,
                         edge_attr=self.target_graph.edge_attr)
        
        return h_src, h_tgt
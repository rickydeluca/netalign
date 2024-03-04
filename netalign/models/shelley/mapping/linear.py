import torch.nn as nn
import torch.nn.functional as F


class LinearMapping(nn.Module):
    """
    Class for a base mapping model that refine the input features
    using a linear layer and align them using the groundtruth aligmnets.
    """
    def __init__(self, source_embeddings, target_embeddings, loss_fn, embedding_dim=256, bias=True):
        super(LinearMapping, self).__init__()
        self.source_embeddings = source_embeddings
        self.target_embeddings = target_embeddings
        self.loss_fn = loss_fn
        self.maps = nn.Linear(embedding_dim, embedding_dim, bias=bias)

    def loss(self, source_batch, target_batch):
        source_feats = self.source_embeddings[source_batch]
        target_feats = self.target_embeddings[target_batch]

        source_feats_after_mapping = self.forward(source_feats)

        batch_size = source_feats.shape[0]
        mapping_loss = self.loss_fn(source_feats_after_mapping, target_feats, source_batch, target_batch) / batch_size
        
        return mapping_loss

    def forward(self, source_feats):
        ret = self.maps(source_feats)
        ret = F.normalize(ret, dim=1)
        return ret
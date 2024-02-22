import torch

from netalign.embedding.embedding_model import EmbeddingModel


class DegreeEmbedding(EmbeddingModel):
    """
    Generate node embeddings using the node degrees.
    """

    def __init__(self, mode='single', embedding_dim=None):
        super(DegreeEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, batch):
        """ Get the embeddings for the nodes in the `batch`. """
        return torch.normal(0, 1, size=(batch.num_nodes, self.embedding_dim))
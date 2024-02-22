import torch

from netalign.embedding.embedding_model import EmbeddingModel


class SharedEmbedding(EmbeddingModel):
    """
    Generate node embeddings a single value (1).
    """

    def __init__(self, embedding_dim):
        super(SharedEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, batch):
        """ Get the embeddings for the nodes in the `batch`. """
        return torch.ones((len(batch), self.embedding_dim))
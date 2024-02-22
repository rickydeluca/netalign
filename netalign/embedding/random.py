import torch

from netalign.embedding.embedding_model import EmbeddingModel


class RandomEmbedding(EmbeddingModel):
    """
    Generate random node embedding using the normal distribution.
    """

    def __init__(self, embedding_dim):
        super(RandomEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, batch):
        """ Get the embeddings for the nodes in the `batch`. """
        return torch.normal(0, 1, size=(len(batch), self.embedding_dim))
import torch

from netalign.embedding.embedding_model import EmbeddingModel


class OneHotEmbedding(EmbeddingModel):
    """
    Generate node embeddings using one-hot encoding.
    """

    def __init__(self):
        super(OneHotEmbedding, self).__init__()

    def forward(self, batch):
        """ Get the embeddings for the nodes in the `batch`. """
        return torch.eye(len(batch))
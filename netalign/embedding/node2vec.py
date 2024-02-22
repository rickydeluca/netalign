import torch
from torch import Tensor
from typing import Optional

from netalign.embedding.embedding_model import EmbeddingModel
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

class Node2VecEmbedding(EmbeddingModel):
    """
    Generate node embeddings using the Node2Vec model.
    """

    def __init__(
        self,
        data,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = False
    ):
        super(Node2VecEmbedding, self).__init__()
        self.model = Node2Vec(edge_index=data.edge_index,
                              embedding_dim=embedding_dim,
                              walk_length=walk_length,
                              context_size=context_size,
                              walks_per_node=walks_per_node,
                              p=p,
                              q=q,
                              num_negative_samples=num_negative_samples,
                              num_nodes=num_nodes,
                              sparse=sparse)
        self.data = data

    def forward(self, batch):
        """ Get the embeddings for the nodes in the `batch`. """
        return self.model(batch)
    
    def train_eval(
        self, 
        optim_name=None, 
        epochs=None,
        lr=None,
        batch_size=None,
        num_workers=4,
        shuffle=True,
        device='cpu'
    ):
        
        # DataLoader
        loader = self.model.loader(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        # Define optimizer
        if optim_name.lower() == 'adam':
            optimizer = torch.optim.Adam(list(self.model.parameters()), lr=lr)
        elif optim_name.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=lr)
        else:
            raise ValueError(f"Invalid optimizer for Node2Vec: {optim_name.lower()}.")
        
        # Train & Eval
        for epoch in range(1, epochs+1):
            loss = self.train(loader=loader, optimizer=optimizer, device=device)
            acc = self.test()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    def train(self, loader=None, optimizer=None, device=None):
        self.model.train()
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader):
            optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    @torch.no_grad()
    def test(self):
        self.model.eval()
        z = self.model()
        acc = self.model.test(z[self.data.train_mask], self.data.y[self.data.train_mask],
                        z[self.data.test_mask], self.data.y[self.data.test_mask],
                        max_iter=150)
        return acc
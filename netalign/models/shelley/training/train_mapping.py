import time

import numpy as np
import torch
from torch_geometric.utils import negative_sampling


def train_gnn(model, optimizer, source_graph, target_graph, gt_aligns, loss_fn=None, epochs=100):

    # Sample negative alignments
    neg_aligns = negative_sampling(gt_aligns, force_undirected=True)
    all_aligns = torch.cat((gt_aligns, neg_aligns), dim=1)

    # Define groundtruth labels
    num_pos_aligns = gt_aligns.shape[1]
    num_neg_aligns = neg_aligns.shape[1]
    gt_labels = torch.cat((torch.ones(num_pos_aligns), torch.zeros(num_neg_aligns)))

    # Shuffle alignments and labels
    shuff_indices = torch.randperm(all_aligns.shape[1])
    all_aligns = all_aligns[:, shuff_indices]
    gt_labels = gt_labels[shuff_indices]

    for epoch in range(epochs):
        model.train()

        # Forward
        hs = model(source_graph.x, source_graph.edge_index, source_graph.edge_attr)
        ht = model(target_graph.x, target_graph.edge_index, target_graph.edge_attr)

        # Extract training subsets
        hs_train = hs[all_aligns[0]]
        ht_train = ht[all_aligns[1]]

        # Predict alignment
        pred_aligns = (hs_train * ht_train).sum(dim=-1)
        
        # Loss
        loss = loss_fn(pred_aligns, gt_labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print or log the loss for monitoring
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


    return model
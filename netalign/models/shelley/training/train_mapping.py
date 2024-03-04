import time

import numpy as np
import torch
from torch_geometric.utils import negative_sampling


def train_gnn_with_neg(model, optimizer, gt_aligns, batch_size=8, epochs=100, device='cpu'):
    """
    Train GNN models with negative sampling
    """
    # Sample negative alignments
    neg_aligns = negative_sampling(gt_aligns, force_undirected=True)
    all_aligns = torch.cat((gt_aligns, neg_aligns), dim=1)

    # Define groundtruth labels
    num_pos_aligns = gt_aligns.shape[1]
    num_neg_aligns = neg_aligns.shape[1]
    gt_labels = torch.cat((torch.ones(num_pos_aligns), torch.zeros(num_neg_aligns))).to(device)

    # Mini-batching
    n_iters = len(gt_labels) // batch_size
    assert n_iters > 0, "`batch_size` is too large."
    if (len(gt_labels)) % batch_size > 0:
        n_iters += 1

    print_every = int(n_iters/4) + 1
    total_steps = 0

    for epoch in range(epochs):
        model.train()

        # Shuffle alignments and labels
        shuff_indices = torch.randperm(all_aligns.shape[1])
        shuff_aligns = all_aligns[:, shuff_indices]
        shuff_labels = gt_labels[shuff_indices]

        for iter in range(n_iters):
            source_batch = shuff_aligns[0][iter*batch_size:(iter+1)*batch_size].to(device)
            target_batch = shuff_aligns[1][iter*batch_size:(iter+1)*batch_size].to(device)
            labels_batch = shuff_labels[iter*batch_size:(iter+1)*batch_size].to(device)
            
            # Forward step
            optimizer.zero_grad()
            start_time = time.time()

            # Loss
            loss = model.loss(source_batch, target_batch, labels_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % print_every == 0 and total_steps > 0:
                print("Iter:", '%03d' %iter,
                      "train_loss=", "{:.5f}".format(loss.item()),
                      "time", "{:.5f}".format(time.time()-start_time))
            total_steps += 1

    return model.state_dict()


def train_gnn_no_neg(model, optimizer, gt_aligns, batch_size=8, epochs=100, device='cpu'):
    """
    Train GNN models without negative sampling.
    """

    source_train_nodes = gt_aligns[0]
    target_train_nodes = gt_aligns[1]

    # Mini-batching
    n_iters = len(source_train_nodes) // batch_size
    assert n_iters > 0, "`batch_size` is too large."
    if (len(source_train_nodes) % batch_size > 0):
        n_iters += 1
    print_every = int(n_iters/4) + 1
    total_steps = 0
    n_epochs = epochs

    # --- Map training --- 
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch: {epoch}')
        model.train()

        # Shuffling
        shuff = torch.randperm(len(source_train_nodes))
        shuffled_source_train_nodes = source_train_nodes[shuff]
        shuffled_target_train_nodes = target_train_nodes[shuff]

        for iter in range(n_iters):
            source_batch = shuffled_source_train_nodes[iter*batch_size:(iter+1)*batch_size].to(device)
            target_batch = shuffled_target_train_nodes[iter*batch_size:(iter+1)*batch_size].to(device)
            
            # Forward step
            optimizer.zero_grad()
            start_time = time.time()

            # Loss
            loss = model.loss(source_batch, target_batch)

            # Backward step
            loss.backward()
            optimizer.step()

            if total_steps % print_every == 0 and total_steps > 0:
                print("Iter:", '%03d' %iter,
                      "train_loss=", "{:.5f}".format(loss.item()),
                      "time", "{:.5f}".format(time.time()-start_time))
            total_steps += 1

    return model.state_dict()


def train_gnn(model, optimizer, gt_aligns, batch_size=8, epochs=100, device='cpu'):
    """
    Train GNN models.
    """

    if model.loss_fn.is_contrastive:
        return train_gnn_with_neg(model=model, optimizer=optimizer, gt_aligns=gt_aligns, batch_size=batch_size, epochs=epochs, device=device)
    else:
        return train_gnn_no_neg(model=model, optimizer=optimizer, gt_aligns=gt_aligns, batch_size=batch_size, epochs=epochs, device=device)


def train_linear(model, optimizer, gt_aligns, batch_size=8, epochs=100, device='cpu'):
    """
    Train the LinearMapping model.
    """

    source_train_nodes = gt_aligns[0]
    target_train_nodes = gt_aligns[1]

    # Mini-batching
    n_iters = len(source_train_nodes) // batch_size
    assert n_iters > 0, "`batch_size` is too large."
    if (len(source_train_nodes) % batch_size > 0):
        n_iters += 1
    print_every = int(n_iters/4) + 1
    total_steps = 0
    n_epochs = epochs

    # --- Map training --- 
    for epoch in range(1, n_epochs + 1):
        print(f'Epoch: {epoch}')
        model.train()

        # Shuffling
        shuff = torch.randperm(len(source_train_nodes))
        shuffled_source_train_nodes = source_train_nodes[shuff]
        shuffled_target_train_nodes = target_train_nodes[shuff]

        for iter in range(n_iters):
            source_batch = shuffled_source_train_nodes[iter*batch_size:(iter+1)*batch_size].to(device)
            target_batch = shuffled_target_train_nodes[iter*batch_size:(iter+1)*batch_size].to(device)
            
            # Forward step
            optimizer.zero_grad()
            start_time = time.time()

            # Loss
            loss = model.loss(source_batch, target_batch)

            # Backward step
            loss.backward()
            optimizer.step()

            if total_steps % print_every == 0 and total_steps > 0:
                print("Iter:", '%03d' %iter,
                      "train_loss=", "{:.5f}".format(loss.item()),
                      "time", "{:.5f}".format(time.time()-start_time))
            total_steps += 1

    return model.state_dict()
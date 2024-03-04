import time

import numpy as np
import torch


def train_pale(model, optimizer, edges=None, epochs=100, batch_size=512, device='cpu'):
    """
    Train function for PALE embedding model.
    """
    n_iters = len(edges) // batch_size
    assert n_iters > 0, "`batch_size` is too large."
    if(len(edges) % batch_size > 0):
        n_iters += 1
    print_every = int(n_iters/4) + 1
    total_steps = 0
    n_epochs = epochs
    for epoch in range(1, n_epochs + 1):
        start = time.time()     # Time evaluation
        model.train()
        print("Epoch {0}".format(epoch))
        shuffle_indices = torch.randperm(len(edges))
        shuffled_edges = edges[shuffle_indices]
        for iter in range(n_iters):
            batch_edges = shuffled_edges[iter*batch_size:(iter+1)*batch_size].to(device)
            start_time = time.time()
            optimizer.zero_grad()
            loss, loss0, loss1 = model.loss(batch_edges[:, 0], batch_edges[:,1])
            loss.backward()
            optimizer.step()
            if total_steps % print_every == 0:
                print("Iter:", '%03d' %iter,
                      "train_loss=", "{:.5f}".format(loss.item()),
                      "true_loss=", "{:.5f}".format(loss0.item()),
                      "neg_loss=", "{:.5f}".format(loss1.item()),
                      "time", "{:.5f}".format(time.time()-start_time))
            total_steps += 1
        
        epoch_time = time.time() - start     # Time evaluation
        
    return model.state_dict()
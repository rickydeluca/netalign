import numpy as np

def compute_accuracy(pred, gt):
    n_matched = 0
    for i in range(pred.shape[0]):
        if pred[i].sum() > 0 and np.array_equal(pred[i], gt[i]):
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes

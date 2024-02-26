import numpy as np

from netalign.evaluation.greedy_match import greedy_match


def compute_accuracy(alignment_matrix, gt_mat, matcher='greedy'):
    # Solve matching
    if matcher == 'greedy':
        perm_mat = greedy_match(alignment_matrix)

    # Compute accuracy
    n_matched = 0
    
    for i in range(perm_mat.shape[0]):
        if perm_mat[i].sum() > 0 and np.array_equal(perm_mat[i], gt_mat[i]):
            n_matched += 1 

    n_nodes = (gt_mat==1).sum()

    return n_matched/n_nodes
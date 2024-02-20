import numpy as np
from netalign.evaluation.greedy_match import greedy_match

def compute_accuracy(alignment_matrix, test_dict, matcher='greedy'):
    # Solve matching
    if matcher == 'greedy':
        perm_mat = greedy_match(alignment_matrix)
    
    # Generate groundtruth matrix from test dict
    gt = np.zeros_like(perm_mat)
    for s, t in test_dict.items():
        gt[s, t] = 1
        gt[t, s] = 1

    n_matched = 0
    for i in range(perm_mat.shape[0]):
        if perm_mat[i].sum() > 0 and np.array_equal(perm_mat[i], gt[i]):
            n_matched += 1 

    n_nodes = (gt==1).sum()
    return n_matched/n_nodes
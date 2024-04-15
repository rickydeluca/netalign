#
# Code from: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html
#

import torch


def log_sinkhorn(log_alpha, n_iter, tau):
    """Performs incomplete Sinkhorn normalization to log_alpha.
    By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
    with positive entries can be turned into a doubly-stochastic matrix
    (i.e. its rows and columns add up to one) via the successive row and column
    normalization.

    [1] Sinkhorn, Richard and Knopp, Paul.
    Concerning nonnegative matrices and doubly stochastic
    matrices. Pacific Journal of Mathematics, 1967
    Args:
      log_alpha: 2D tensor (a matrix of shape [N, N])
        or 3D tensor (a batch of matrices of shape = [batch_size, N, N])
      n_iters: number of sinkhorn iterations (in practice, as little as 20
        iterations are needed to achieve decent convergence for N~100)
    Returns:
      A 3D tensor of close-to-doubly-stochastic matrices (2D tensors are
        converted to 3D tensors with batch_size equals to 1)
    """
    # Temperature scaling
    log_alpha = log_alpha / tau

    # Sinkhorn iterations
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)

    return log_alpha.exp()


def mask_indices(tensor, row_indices, col_indices):
    batch_size, N, M = tensor.size()
    mask = torch.zeros(batch_size, N, M, dtype=torch.bool, device=tensor.device)
    mask[:, row_indices, :] = True
    mask[:, :, col_indices] = True
    mask[:, :, [i for i in range(M) if i not in col_indices]] = False  # Ensure only specified columns are kept
    mask[:, [i for i in range(M) if i not in row_indices], :] = False # Ensure only specified rows are kept
    return tensor.masked_fill(~mask, 1e-20)

if __name__ == '__main__':
    device = torch.device('cuda:0')

    # Toy similarity matrix
    sim_mat = torch.randn((1,5,5), device=device)

    # Rescale to positive
    min_value = torch.min(sim_mat)
    sim_mat = sim_mat + torch.abs(min_value) + 1

    # Extract submatrix
    row_indices = [0, 1, 2]
    col_indices = [0, 3, 4]
    sim_mat = mask_indices(sim_mat, row_indices, col_indices)
    sim_mat = torch.log(sim_mat)

    # Sinkhorn normalization
    rank_mat = log_sinkhorn(sim_mat, tau=1.0, n_iter=100)

    print('sim_mat:\n', sim_mat)
    print('-----------------'*5)
    print('rank_mat:\n', rank_mat)

    # Check if the result is doubly stochastic
    print("Row sum:", torch.sum(rank_mat, dim=1))
    print("Col sum:", torch.sum(rank_mat, dim=2))

    # print("\n Top K rank values:", torch.topk(rank_mat, k=5, dim=2)[0])

from multiprocessing import Pool

import numpy as np
import torch
from torch import Tensor

from netalign.models.shelley.utils.lap_solvers import log_sinkhorn
from netalign.models.shelley.utils.sm_solvers.stable_marriage_data_processor import \
    SMDataProcessor


def stable_marriage(s: Tensor, n1: Tensor=None, n2: Tensor=None, nproc: int=1) -> Tensor:
    r"""
    :param s: :math:`(b\times n_1 \times n_2)` input 3d tensor. :math:`b`: batch size
    :param n1: :math:`(b)` number of objects in dim1
    :param n2: :math:`(b)` number of objects in dim2
    """
    if len(s.shape) == 2:
        s = s.unsqueeze(0)
        matrix_input = True
    elif len(s.shape) == 3:
        matrix_input = False
    else:
        raise ValueError('input data shape not understood: {}'.format(s.shape))

    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy()
    if n1 is not None:
        n1 = n1.cpu().numpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.cpu().numpy()
    else:
        n2 = [None] * batch_num

    if nproc > 1:
        print("nproc GREATER than 1")
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_stable_marriage, zip(perm_mat, n1, n2))
            
            output=mapresult.get()
            perm_mat = np.stack(perm_mat,output['perm_mat'])
            attention_mat = np.stack(perm_mat,output['attention_mat'])

    else:
        
        perm_mat_list=[]
        attention_mat_list=[]
        for b in range(batch_num):
            outputs = _stable_marriage(perm_mat[b], n1[b], n2[b])
            perm_mat_list.append(outputs['perm_mat'])
            attention_mat_list.append(outputs['attention_mat'])
            
           
        perm_mat_new = np.stack( perm_mat_list, axis=0 )
        attention_mat = np.stack( attention_mat_list, axis=0 )
        
    perm_mat_new=torch.from_numpy(perm_mat_new).to(device)
    attention_mat=torch.from_numpy(attention_mat).to(device)
    if matrix_input:
        perm_mat_new.squeeze_(0)
        attention_mat.squeeze_(0)
    
    return perm_mat_new

def _stable_marriage(s: torch.Tensor, n1=None, n2=None):
    if n1 is None:
        n1 = s.shape[0]
    if n2 is None:
        n2 = s.shape[1]
    raw=s
    
    row_difference=s.shape[0]-n1
    column_difference=s.shape[1]-n2
    row_mat_np = s[:n1, :n2]
    column_mat = []
    for i in range(n2):
        column_mat.append(row_mat_np[:, i])
    column_mat_np = np.array(column_mat)
    smdp = SMDataProcessor(row_mat_np,column_mat_np)
    row, col = smdp.get_row_col_of_stable_marriage_permutation()
    perm_mat = np.zeros_like(s)
    perm_mat[row, col] = 1
    initial_attention_mat = smdp.get_stable_marriage_attention()
    attention_mat= np.pad(initial_attention_mat, [(0, row_difference), (0, column_difference)], mode='constant', constant_values=0)
    
    return {'perm_mat': perm_mat, 'attention_mat': attention_mat}

if __name__ == '__main__':
    
    # Toy similarity matrix
    device = torch.device('cuda:0')
    sim_mat = torch.rand((1, 100, 100), device=device)
    print('sim_mat:', sim_mat)

    # Ranking matrix
    rank_mat = log_sinkhorn(sim_mat, n_iter=20)
    print('rank_mat:', rank_mat)

    # Stable marriage algorithm
    n_src = torch.tensor(rank_mat.size(1)).unsqueeze(0)
    n_tgt = torch.tensor(rank_mat.size(2)).unsqueeze(0)
    pred_mat = stable_marriage(rank_mat, n_src, n_tgt)
    
    print('pred_mat:', pred_mat)
    print('Unique predicted values:', torch.unique(pred_mat))
    
    # Check if the result is a permutation matrix
    print("Row sum:", torch.sum(pred_mat, dim=1))
    print("Col sum:", torch.sum(pred_mat, dim=2))
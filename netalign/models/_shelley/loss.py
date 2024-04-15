import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()
        self.loss_fn = self._euclidean_loss

    def forward(self, inputs1, inputs2):
        return self.loss_fn(inputs1, inputs2)

    def _euclidean_loss(self, inputs1, inputs2):
        sub = inputs2 - inputs1
        square_sub = sub**2
        loss = torch.sum(square_sub)        
        return loss


class PermutationLoss(nn.Module):
    def __init__(self):
        super(PermutationLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor) -> Tensor:
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)
        gt_perm = gt_perm.to(dtype=torch.float32)
        
        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
            loss += F.binary_cross_entropy(
                pred_dsmat[batch_slice],
                gt_perm[batch_slice],
                reduction='sum')
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum
    

class ContrastiveLossWithAttention(nn.Module):
    """
    Contrastive loss with attention between two permutations,
    as described in "Contrastive learning for supervised graph matching"
    by Ratnayaka et al. (2023).
    """
    def __init__(self):
        super(ContrastiveLossWithAttention, self).__init__()
    
    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor, beta_value) -> Tensor:
        
        batch_num = pred_dsmat.shape[0]
      
        pred_dsmat = torch.clamp(pred_dsmat, min=0.0, max=1.0)
        gt_predicted_values = torch.mul(pred_dsmat, gt_perm)
        
        beta = torch.full_like(pred_dsmat, beta_value)
        
        column_gt_values = torch.matmul(torch.ones_like(gt_predicted_values), gt_predicted_values)
        column_gt_values_minus_beta = column_gt_values-beta
        gt_available_columns = torch.matmul(torch.ones_like(gt_predicted_values), gt_perm)
        
        attention_tgt = torch.mul(torch.ge(pred_dsmat,column_gt_values_minus_beta).float(), gt_available_columns)
        attention_tgt_without_gt = attention_tgt - gt_perm
        attention_tgt_predicted_values_without_gt = torch.mul(attention_tgt_without_gt, pred_dsmat)  
        attention_tgt_negatives_selected = attention_tgt_predicted_values_without_gt
        
        row_gt_values = torch.matmul(gt_predicted_values, torch.ones_like(gt_predicted_values))
        row_gt_values_minus_beta = row_gt_values - beta
        gt_available_rows = torch.matmul(gt_perm, torch.ones_like(gt_predicted_values))
       
        attention_src = torch.mul(torch.ge(pred_dsmat, row_gt_values_minus_beta).float(), gt_available_rows)
        attention_src_without_gt = attention_src - gt_perm
        attention_src_predicted_values_without_gt = torch.mul(attention_src_without_gt, pred_dsmat)
        attention_src_negatives_selected = attention_src_predicted_values_without_gt

        def calculateLoss(
            gt_perm,
            gt_predicted_values,
            attention_src_negatives_selected,
            attention_tgt_negatives_selected,
            mask=False
        ):
                    
            gt_indices = (gt_perm == 1).nonzero(as_tuple=True)
            corresponding_target_indices = gt_indices[1]
            corresponding_source_indices = gt_indices[0]
            
            attention_src_negatives_selected_squared = torch.square(attention_src_negatives_selected)
            attention_tgt_negatives_selected_squared = torch.square(attention_tgt_negatives_selected)
            gt_predicted_values_squared = torch.square(gt_predicted_values)

            src_negative_sum = torch.sum(attention_src_negatives_selected_squared,1)
            src_positive_sum = torch.sum(gt_predicted_values_squared,1)
            tgt_negative_sum = torch.sum(attention_tgt_negatives_selected_squared,0)

            corresponding_tgt_negative_sum = torch.index_select(tgt_negative_sum, 0, corresponding_target_indices)
            
            if mask is True:
                src_negative_sum = torch.index_select(src_negative_sum, 0, corresponding_source_indices) 
                src_positive_sum = torch.index_select(src_positive_sum, 0, corresponding_source_indices) 

            overall_negative_sum = src_negative_sum + corresponding_tgt_negative_sum 
            
            denominator = 1 + overall_negative_sum
            probability = src_positive_sum/denominator
            elementwise_loss = -0.5 * torch.log(probability)
            
            loss = torch.sum(elementwise_loss)
            return loss
            
        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)

        for b in range(batch_num):
            loss += calculateLoss(
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                gt_predicted_values[b, :src_ns[b], :tgt_ns[b]],
                attention_src_negatives_selected[b, :src_ns[b], :tgt_ns[b]],
                attention_tgt_negatives_selected[b, :src_ns[b], :tgt_ns[b]],
            )

            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
         
        return loss/n_sum
    

class RandomContrastiveLossWithAttention(nn.Module):
    """
    Contrastive loss with attention between two permutations,
    as described in "Contrastive learning for supervised graph matching"
    by Ratnayaka et al. (2023).
    """
    def __init__(self):
        super(RandomContrastiveLossWithAttention, self).__init__()

    @staticmethod
    def keep_top_p_values(x, dim=-1, p=0.1):
        """
        Select for the specifed dimension the top `p`
        values, where `p` is in percentage.

        Code adapted from: 
        https://discuss.pytorch.org/t/how-to-keep-only-top-k-percent-values/83706
        """
        orig_shape = x.size()

        # Calculate positions of top p values
        num_values = x.size(dim)
        ret = torch.topk(x, k=int(p*num_values), dim=dim)
    
        # Scatter to zero'd tensor
        res = torch.zeros_like(x)
        res.scatter_(dim, ret.indices, ret.values)
        res = res.view(*orig_shape)
        return res
    
    @staticmethod
    def keep_top_k_values(x, dim=-1, k=5):
        """
        Select for the specifed dimension the top `k`
        values, where `k` is the number of nodes

        Code adapted from: 
        https://discuss.pytorch.org/t/how-to-keep-only-top-k-percent-values/83706
        """
        orig_shape = x.size()

        # Calculate positions of top p values
        ret = torch.topk(x, k=k, dim=dim)
    
        # Scatter to zero'd tensor
        res = torch.zeros_like(x)
        res.scatter_(dim, ret.indices, ret.values)
        res = res.view(*orig_shape)
        return res
    
    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor, top_k=5) -> Tensor:
        
        batch_num = pred_dsmat.shape[0]
      
        pred_dsmat = torch.clamp(pred_dsmat, min=0.0, max=1.0)
        gt_predicted_values = torch.mul(pred_dsmat, gt_perm)
        
        column_gt_values = torch.matmul(torch.ones_like(gt_predicted_values), gt_predicted_values)
        gt_available_columns = torch.matmul(torch.ones_like(gt_predicted_values), gt_perm)
        
        attention_tgt = torch.mul(self.keep_top_k_values(column_gt_values, dim=1, k=top_k).float(), gt_available_columns)
        attention_tgt_without_gt = attention_tgt - gt_perm
        attention_tgt_predicted_values_without_gt = torch.mul(attention_tgt_without_gt, pred_dsmat)  
        attention_tgt_negatives_selected = attention_tgt_predicted_values_without_gt
        
        row_gt_values = torch.matmul(gt_predicted_values, torch.ones_like(gt_predicted_values))
        gt_available_rows = torch.matmul(gt_perm, torch.ones_like(gt_predicted_values))
       
        attention_src = torch.mul(self.keep_top_k_values(row_gt_values, dim=2, k=top_k).float(), gt_available_rows)
        attention_src_without_gt = attention_src - gt_perm
        attention_src_predicted_values_without_gt = torch.mul(attention_src_without_gt, pred_dsmat)
        attention_src_negatives_selected = attention_src_predicted_values_without_gt

        def calculateLoss(
            gt_perm,
            gt_predicted_values,
            attention_src_negatives_selected,
            attention_tgt_negatives_selected,
            mask=False
        ):
                    
            gt_indices = (gt_perm == 1).nonzero(as_tuple=True)
            corresponding_target_indices = gt_indices[1]
            corresponding_source_indices = gt_indices[0]
            
            attention_src_negatives_selected_squared = torch.square(attention_src_negatives_selected)
            attention_tgt_negatives_selected_squared = torch.square(attention_tgt_negatives_selected)
            gt_predicted_values_squared = torch.square(gt_predicted_values)

            src_negative_sum = torch.sum(attention_src_negatives_selected_squared,1)
            src_positive_sum = torch.sum(gt_predicted_values_squared,1)
            tgt_negative_sum = torch.sum(attention_tgt_negatives_selected_squared,0)

            corresponding_tgt_negative_sum = torch.index_select(tgt_negative_sum, 0, corresponding_target_indices)
            
            if mask is True:
                src_negative_sum = torch.index_select(src_negative_sum, 0, corresponding_source_indices) 
                src_positive_sum = torch.index_select(src_positive_sum, 0, corresponding_source_indices) 

            overall_negative_sum = src_negative_sum + corresponding_tgt_negative_sum 
            
            denominator = 1 + overall_negative_sum
            probability = src_positive_sum/denominator
            elementwise_loss = -0.5 * torch.log(probability)
            
            loss = torch.sum(elementwise_loss)
            return loss
            
        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)

        for b in range(batch_num):
            loss += calculateLoss(
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                gt_predicted_values[b, :src_ns[b], :tgt_ns[b]],
                attention_src_negatives_selected[b, :src_ns[b], :tgt_ns[b]],
                attention_tgt_negatives_selected[b, :src_ns[b], :tgt_ns[b]],
            )

            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
         
        return loss/n_sum
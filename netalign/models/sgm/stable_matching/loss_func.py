import torch
import torch.nn as nn
from torch import Tensor

class ContrastiveLossWithAttention(nn.Module):
    def __init__(self):
        super(ContrastiveLossWithAttention, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor, beta_value: float) -> Tensor:
        pred_dsmat = torch.clamp(pred_dsmat, min=0.0, max=1.0)
        batch_num = pred_dsmat.shape[0]

        beta = torch.full_like(pred_dsmat, beta_value)
        all_zeros_tensor = torch.zeros_like(gt_perm)

        loss = torch.tensor(0.0).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)

        for b in range(batch_num):
            gt_perm_b = gt_perm[b, :src_ns[b], :tgt_ns[b]]
            pred_dsmat_b = pred_dsmat[b, :src_ns[b], :tgt_ns[b]]

            gt_predicted_values = pred_dsmat_b * gt_perm_b
            row_gt_values = torch.sum(gt_predicted_values, dim=1)
            column_gt_values = torch.sum(gt_predicted_values, dim=0)

            attention_src = (pred_dsmat_b >= row_gt_values - beta).float() * torch.sum(gt_perm_b, dim=1)
            attention_tgt = (pred_dsmat_b >= column_gt_values - beta).float() * torch.sum(gt_perm_b, dim=0)

            attention_src_without_gt = attention_src - gt_perm_b
            attention_tgt_without_gt = attention_tgt - gt_perm_b

            attention_src_negatives_selected = attention_src_without_gt * pred_dsmat_b
            attention_tgt_negatives_selected = attention_tgt_without_gt * pred_dsmat_b

            corresponding_target_indices = (gt_perm_b == 1).nonzero(as_tuple=True)[1]

            src_negative_sum = torch.sum(attention_src_negatives_selected**2, dim=1)
            src_positive_sum = torch.sum(gt_predicted_values**2, dim=1)

            tgt_negative_sum = torch.sum(attention_tgt_negatives_selected**2, dim=0)
            corresponding_tgt_negative_sum = torch.sum(tgt_negative_sum[corresponding_target_indices])

            overall_negative_sum = src_negative_sum + corresponding_tgt_negative_sum
            denominator = 1 + overall_negative_sum
            probability = src_positive_sum / denominator
            elementwise_loss = -0.5 * torch.log(probability)

            loss += torch.sum(elementwise_loss)
            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

        return loss / n_sum

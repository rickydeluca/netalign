import torch
import torch.nn as nn
from netalign.models.shelley.mapping.loss.loss import MappingLoss

class EuclideanLoss(MappingLoss):
    def __init__(self):
        super(MappingLoss, self).__init__()
        self.is_contrastive = False             # Used in training to sample neg aligmnets
        self.loss_fn = self._euclidean_loss

    def forward(self, source_feats, target_feats, source_indices=None, target_indices=None, gt_labels=None):
        loss = self.loss_fn(source_feats, target_feats, source_indices, target_indices, gt_labels)
        return loss

    def _euclidean_loss(self, inputs1, inputs2):
        sub = inputs2 - inputs1
        square_sub = sub**2
        loss = torch.sum(square_sub)        
        return loss
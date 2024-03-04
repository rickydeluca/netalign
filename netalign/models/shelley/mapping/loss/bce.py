import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.is_contrastive = True     # Used in training to sample neg aligmnets
        self.loss_fn = nn.BCEWithLogitsLoss(weight=weight,
                                            size_average=size_average,
                                            reduce=reduce,
                                            reduction=reduction,
                                            pos_weight=pos_weight)
        
    def forward(self, source_feats, target_feats, source_indices, target_indices, gt_labels):
        pred = (source_feats * target_feats).sum(dim=-1)
        loss = self.loss_fn(pred, gt_labels)
        return loss
    

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.is_contrastive = True
        self.loss_fn = nn.BCELoss(weight=weight,
                                  size_average=size_average,
                                  reduce=reduce,
                                  reduction=reduction)
        
    def forward(self, source_feats, target_feats, source_indices, target_indices, gt_labels):
        pred = (source_feats * target_feats).sum(dim=-1)
        loss = self.loss_fn(pred, gt_labels)
        return loss

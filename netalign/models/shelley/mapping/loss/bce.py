import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(weight=weight,
                                            size_average=size_average,
                                            reduce=reduce,
                                            reduction=reduction,
                                            pos_weight=pos_weight)
        
    def forward(self, input, target):
        loss = self.loss_fn(input, target)
        return loss
    

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCELoss(weight=weight,
                                  size_average=size_average,
                                  reduce=reduce,
                                  reduction=reduction)
        
    def forward(self, input, target):
        loss = self.loss_fn(input, target)
        return loss

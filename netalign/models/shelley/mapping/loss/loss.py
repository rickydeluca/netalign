import torch.nn as nn


class MappingLoss(nn.Module):
    """
    Base class for the mapping loss functions
    """
    def __init__(self):
        super(MappingLoss, self).__init__()
        self.is_contrastive = False         # Used in training to sample neg aligmnets
        self.loss_fn = None

    def forward(self, source_feats, target_feats, input_indices, target_indices):
        return
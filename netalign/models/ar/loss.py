import torch


class MappingLossFunctions(object):
    def __init__(self):
        self.loss_fn = self._euclidean_loss

    def loss(self, inputs1, inputs2):
        return self.loss_fn(inputs1, inputs2)

    def _euclidean_loss(self, inputs1, inputs2):
        sub = inputs2 - inputs1
        square_sub = sub**2
        loss = torch.sum(square_sub)        
        return loss
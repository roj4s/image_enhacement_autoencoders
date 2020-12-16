import torch.nn as nn
import contextual_loss as cl

class ContextualPlusMse(nn.Module):

    def __init__(self, _lambda=0.1, device='cuda:0'):
        super(ContextualPlusMse, self).__init__()
        self._lambda = 0.1
        self.device = device
        self.mse_criterion = nn.MSELoss().to(device)
        self.cx_criterion = cl.ContextualLoss(use_vgg=True,
                                         vgg_layer='relu5_4').to(self.device)


    def forward(self, y, y_hat):
        l1_loss = self._lambda * self.mse_criterion(y, y_hat)
        cx_loss = (1. - self._lambda) * self.cx_criterion(y, y_hat)
        return cx_loss * l1_loss

from args import args
import contextual_loss as cl
import torch.nn as nn
import torch

device_ids = range(torch.cuda.device_count())
dev = f'cuda:{device_ids[-1]}'

loss_functions = {
    'cobi': cl.ContextualBilateralLoss(use_vgg=True,
                                       vgg_layer='relu5_4').to(dev),
    'mse': nn.MSELoss(),
    'contextual_loss': cl.ContextualLoss(use_vgg=True,
                                         vgg_layer='relu5_4').to(dev)
}

loss_str = ','.join(list(loss_functions.keys()))

if args.loss not in loss_functions:
    raise f"Invalid loss function. Must be one of: {loss_str}"

loss = loss_functions[args.loss]



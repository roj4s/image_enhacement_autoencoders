from args import args
import contextual_loss as cl
import torch.nn as nn
import torch
from contextual_plus_mse_loss import ContextualPlusMse

device_ids = range(torch.cuda.device_count())
dev = f'cuda:{device_ids[-1]}'

loss_function_names = set(['l1', 'contextual_loss', 'contextual_plus_l1'])

loss_str = ','.join(list(loss_function_names))

if args.loss not in loss_functions:
    raise f"Invalid loss function. Must be one of: {loss_str}"

if args.loss == 'l1':
    loss = nn.MSELoss()

else if args.loss == 'contextual_loss':
    loss = cl.ContextualLoss(use_vgg=True,
                                         vgg_layer='relu5_4').to(dev)

else if args.loss == 'contextual_plus_l1':
    loss = ContextualPlusMse(_lambda=args.cx_plus_l1_lambda, device=dev).to(dev)

loss = loss_functions[args.loss]



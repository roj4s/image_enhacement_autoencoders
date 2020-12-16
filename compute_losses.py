from data import EnumPairedDataset
from args import args
from loss import loss
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import os

dataset = EnumPairedDataset(args.train_x_root, args.train_y_root)
dl = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

columns = ['file_name', args.loss]
cols_str = ",".join(columns)
if not os.path.exists(args.logs):
    os.makedirs(args.logs)
logs_output = os.path.join(args.logs, f'{args.loss}.csv')

with open(logs_output, 'wt') as f:
    f.write(f"{cols_str}\n")


with torch.no_grad():
    for i, data in tqdm(enumerate(dl), total=int(len(dataset)
                                / 2)):
        x, y, fn = data
        l = loss(x, y).item()
        with open(logs_output, 'at') as f:
            f.write(f"{fn[0]},{l}\n")

from data import EnumPairedDataset
from data import SingleFolderDataset
import piq
import pandas as pd
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

def image_metrics_from_dataset(dataset, output_addr='/tmp/psnr.csv'):

    cols = ['file_name', 'psnr', 'ssim']
    cols_str = ','.join(cols)

    with open(output_addr, 'wt') as f:
        f.write(f'{cols_str}\n')

    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data in tqdm(enumerate(dl), total=int(len(dataset))):
        x, y, file_name = data
        psnr = piq.psnr(x, y).item()
        ssim = piq.ssim(x, y).item()
        vals = [file_name[0], str(psnr), str(ssim)]
        val_str = ','.join(vals)
        with open(output_addr, 'at') as f:
            f.write(f'{val_str}\n')

    return pd.read_csv(output_addr)

if __name__ == "__main__":
    #root = '/home/rojas/datasets/real-world-super-resolution/Train_x2/'
    root = '/home/rojas/dev/autoencoder_image_enhacement/resnet_autoencoder/images/epoch_200'
    #x_root = os.path.join(root, 'train_LR_bicubic_from_HR')
    #y_root = os.path.join(root, 'train_LR')
    #dataset = EnumPairedDataset(x_root, y_root)
    dataset = SingleFolderDataset(root)
    image_metrics_from_dataset(dataset, output_addr=os.path.join(root,
                                                                 'metrics.csv'))

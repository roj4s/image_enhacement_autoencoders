import argparse
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser(description='Train flow for cnn '\
                                 'degradation model autoencoder')
parser.add_argument('train_x_root', type=str, help="Directory containing input "\
                    "images")
parser.add_argument("train_y_root", type=str, help="Directory containing golden "\
                    "images")
parser.add_argument('--test-x', type=str, help="Directory containing input test "\
                    "images", default=None)
parser.add_argument("--test-y", type=str, help="Directory containing test golden "\
                    "images", default=None)
parser.add_argument('--scale', type=str, help="Scale", default='x2')
parser.add_argument('--model', type=str, help="Model name",
                    default='resnet_autoencoder')
parser.add_argument('--epochs', type=int, help="Epochs to train model, "\
                    "default is 200",
                    default=200)
parser.add_argument('--checkpoints-output', help='Optional, directory to '\
                    'output checkpoints', type=str,
                    default=f'/tmp/checkpoints_{timestamp}')
parser.add_argument('--logs', help='Optional, directory to '\
                    'output logs (e.g metrics, images)', type=str,
                    default=f'/tmp/train_logs_{timestamp}')
parser.add_argument('--show-dataset', action='store_true',
                help='Plot dataset samples', default=False)
parser.add_argument('--learning-rate', type=float,
                help='Learning rate to use with Adam optimizer, default is \
                    1e-3', default=1e-3)
parser.add_argument('--show-output-images', action='store_true',
                help='Whether tensorboard/pytorch should save output images after '\
                    'each epoch', default=False)
parser.add_argument('--device', help='Device to use for training, default to '\
                    'CPU', type=str,
                    default='cpu')
parser.add_argument('--loss', help='Loss function, default to MSE', type=str,
                    default='mse')

args = parser.parse_args()
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

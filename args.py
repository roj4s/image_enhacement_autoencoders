import argparse
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
parser = argparse.ArgumentParser(description='Train flow for cnn '\
                                 'degradation model autoencoder')
parser.add_argument('x_root', type=str, help="Directory containing input "\
                    "images")
parser.add_argument("y_root", type=str, help="Directory containing golden "\
                    "images")
parser.add_argument('--scale', type=str, help="Scale", default='x2')
parser.add_argument('--model', type=str, help="Model name", default='Autoencoder_x2_s')
parser.add_argument('--epochs', type=int, help="Epochs to train model, "\
                    "default is 200",
                    default=200)
parser.add_argument('--checkpoints-output', help='Optional, directory to '\
                    'output checkpoints', type=str,
                    default=f'/tmp/checkpoints_{timestamp}')
parser.add_argument('--tensorboard-output', help='Optional, directory to '\
                    'output tensorboard logs', type=str,
                    default=f'/tmp/tensorboard_{timestamp}')
parser.add_argument('--show-dataset', action='store_true',
                help='Plot dataset samples', default=False)
parser.add_argument('--learning-rate', type=float,
                help='Learning rate to use with Adam optimizer, default is \
                    1e-3', default=1e-3)
parser.add_argument('--show-output-images', action='store_true',
                help='Whether tensorboard should show output images after '\
                    'each epoch', default=False)

args = parser.parse_args()
for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

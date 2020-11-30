from resnet_autoencoder import ResNet_autoencoder
from resnet_autoencoder import Bottleneck
from resnet_autoencoder import DeconvBottleneck

models = {
    'resnet_autoencoder': ResNet_autoencoder(Bottleneck, DeconvBottleneck, [
        1, 4, 6, 3], 3)
}

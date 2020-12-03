from torch import nn
import torch
import torchvision.models as models

class Resnet50FeaturesExtractor(nn.Module):

    def __init__(self):
        super(Resnet50FeaturesExtractor, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True, progress=True)
        self.res50_conv = torch.nn.Sequential(*list(self.resnet50.children())[:-2])
        for param in self.res50_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.res50_conv(x)

if __name__ == "__main__":
    _input = torch.autograd.Variable(torch.randn(2, 3, 380, 380)).cpu()
    res50_conv = Resnet50FeaturesExtractor()
    o = res50_conv(_input)
    print(o.shape)


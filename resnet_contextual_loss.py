import torch





def contextual_loss(x, y, h=0.5):
    """Computes contextual loss between x and y.
        Taken from:
        https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da

    Args:
      x: features of shape (N, C, H, W).
      y: features of shape (N, C, H, W).

    Returns:
      cx_loss = contextual loss between x and y (Eq (1) in the paper)
    """
    assert x.size() == y.size()
    N, C, H, W = x.size()   # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

    y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)

    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

    # The equation at the bottom of page 6 in the paper
    # Vectorized computation of cosine similarity for each pair of x_i and y_j
    x_normalized = x_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)                                # (N, C, H*W)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)           # (N, H*W, H*W)

    d = 1 - cosine_sim                                  # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data 
    d_min, _ = torch.min(d, dim=2, keepdim=True)        # (N, H*W, 1)

    # Eq (2)
    d_tilde = d / (d_min + 1e-5)

    # Eq(3)
    w = torch.exp((1 - d_tilde) / h)

    # Eq(4)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)       # (N, H*W, H*W)

    # Eq (1)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
    cx_loss = torch.mean(-torch.log(cx + 1e-5))

    return cx_loss

class Resnet50ContextualLoss(torch.nn.Module):

    def __init__(self, device='cpu'):
        super(Resnet50ContextualLoss, self).__init__()
        self.res50_conv = Resnet50FeaturesExtractor().to(device)
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def forward(self, y, y_hat):
        '''
        y = self.transform(y)
        y_hat = self.transform(y_hat)
        y_ft = self.res50_conv(y)
        y_hat_ft = self.res50_conv(y_hat)
        '''
        return contextual_loss(y, y_hat)

if __name__ == "__main__":
    from resnet_features_extractor import Resnet50FeaturesExtractor
    from data import SingleFolderDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torchvision import transforms
    import numpy as np
    import sys

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    y = torch.autograd.Variable(torch.randn(32, 3, 380, 380)).to(sys.argv[1])
    y_hat = torch.autograd.Variable(torch.randn(32, 3, 380, 380)).to(sys.argv[1])
    cx = Resnet50ContextualLoss(device=sys.argv[1]).to(sys.argv[1])
    l = cx(y, y_hat)
    print(l)

    '''
    toTensor = transforms.ToTensor()
    root = sys.argv[1]
    dataset = SingleFolderDataset(root)
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, data in tqdm(enumerate(dl), total=int(len(dataset))):
        x, y, file_name = data
        yi = utils.tensor_to_image(y)
        print(yi.shape)
        yi = yi[5:, 5:, :]
        yi = np.reshape(yi, (-1, yi.shape[0], yi.shape[1], yi.shape[2]))
        yi = utils.image_to_tensor(yi)
        y = normalize(y)
        yi = normalize(yi)
        o = res50_conv(y)
        oi = res50_conv(yi)
        l = contextual_loss(o, oi)
        print(l.item())
    '''

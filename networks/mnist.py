import torch
import torch.nn as nn
from torch.nn.functional import softmax

class MNIST_NET(nn.Module):
    def __init__(self, N=10):
        super(MNIST_NET, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
        )

    def forward(self, x, ratio=None, tau=1, do_softmax=True, eval_only = True):
        feat = self.encoder(x)
        x = feat.view(-1, 16 * 4 * 4)
        x = self.fc(x)
        if ratio is not None:
            ratio = ratio.reshape(x.shape[-1]).to(x.device)
            x -= tau * torch.log(ratio)
        if do_softmax:
            x = softmax(x, dim=1)
        if eval_only:
            return x
        else:
            return x, feat
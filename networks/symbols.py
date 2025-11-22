import torch
from torch import nn
import torch.nn.functional as F

class SymbolNet(nn.Module):
    def __init__(self):
        super(SymbolNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(30976, 128)
        self.fc2 = nn.Linear(128, 13) # 0--9, +, -, *

    def forward(self, x, ratio=None, tau=1, do_softmax=True, eval_only = True):
        x = x.float()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        feat = torch.flatten(x, 1)
        x = self.fc1(feat)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        if ratio is not None:
            ratio = ratio.reshape(x.shape[-1]).to(x.device)
            x -= tau * torch.log(ratio)
        if do_softmax:
            x = F.softmax(x, dim=1)
        if eval_only:
            return x
        else:
            return x, feat
        
        #return F.softmax(x, dim=1)
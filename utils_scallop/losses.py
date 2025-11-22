import torch
import torch.nn.functional as F


def bce_loss(output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack(
        [
            torch.tensor([1.0 if i == t else 0.0 for i in range(dim)])
            for t in ground_truth
        ]
    ).cuda()
    return F.binary_cross_entropy(output, gt)


def nll_loss(output, ground_truth):
    return F.nll_loss(output, ground_truth)

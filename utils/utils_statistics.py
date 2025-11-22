import numpy as np
import torch
import math


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = (
            eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        )
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_partial(outputs, target, transition):
    """
    Computes the accuracy of the predictions for the partial label
    """
    with torch.no_grad():
        M = len(outputs)
        batch_size = target.size(0)

        pred = [output.argmax(dim=1) for output in outputs]
        pred = [transition([pred[i][j] for i in range(M)]) for j in range(batch_size)]
        pred = torch.tensor(pred).view_as(target)

        correct = pred.eq(target)
        correct = correct.float().sum(0, keepdim=True)
        acc = correct * 100.0 / batch_size
        return acc, correct, batch_size


def accuracy_partial_cifar(outputs, target, transition):
    """
    Computes the accuracy of the predictions for the partial label
    """
    with torch.no_grad():
        batch_size = len(target)

        pred = [output.argmax(dim=1) for output in outputs]
        pred = [transition(pred[0][j].item(), pred[1][j].item()) for j in range(batch_size)]    # assuming M=2
        
        correct = sum(pred[i] == target[i] for i in range(len(pred)))

        return correct, batch_size

class AccurracyShot(object):
    def __init__(self, train_class_count, num_class, many_shot_num=3, low_shot_num=3):
        self.train_class_count = train_class_count
        self.test_class_count = None
        self.num_class = num_class
        self.many_shot_thr = train_class_count.sort()[0][num_class - many_shot_num - 1]
        self.low_shot_thr = train_class_count.sort()[0][low_shot_num]

    def get_shot_acc(self, preds, labels, acc_per_cls=False):
        if self.test_class_count is None:
            self.test_class_count = []
            for l in range(self.num_class):
                self.test_class_count.append(len(labels[labels == l]))

        class_correct = []
        for l in range(self.num_class):
            class_correct.append((preds[labels == l] == labels[labels == l]).sum())

        many_shot = []
        median_shot = []
        low_shot = []
        for i in range(self.num_class):
            if self.train_class_count[i] > self.many_shot_thr:
                many_shot.append((class_correct[i] / float(self.test_class_count[i])))
            elif self.train_class_count[i] < self.low_shot_thr:
                low_shot.append((class_correct[i] / float(self.test_class_count[i])))
            else:
                median_shot.append((class_correct[i] / float(self.test_class_count[i])))

        if len(many_shot) == 0:
            many_shot.append(0)
        if len(median_shot) == 0:
            median_shot.append(0)
        if len(low_shot) == 0:
            low_shot.append(0)

        if acc_per_cls:
            class_accs = [
                c / cnt for c, cnt in zip(class_correct, self.test_class_count)
            ]
            return (
                np.mean(many_shot) * 100,
                np.mean(median_shot) * 100,
                np.mean(low_shot) * 100,
                class_accs,
            )
        else:
            return (
                np.mean(many_shot) * 100,
                np.mean(median_shot) * 100,
                np.mean(low_shot) * 100,
            )


def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples


def sigmoid_rampup(current, rampup_length, exp_coe=5.0):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-exp_coe * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(0.5 * (np.cos(np.pi * current / rampdown_length) + 1))

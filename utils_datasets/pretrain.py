import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .randaugment import RandomAugment
import copy
from resnet import resnet18

def get_accuracy(model, dataloader: DataLoader):
    total = 0
    correct = 0
    for input_data, gt_labels in dataloader:
        _, predicted = torch.max(model(input_data), 1)
        total += len(gt_labels)
        correct_labels = torch.eq(predicted, gt_labels)
        correct += correct_labels.sum().item()
    return correct / total

accuracies = dict()

def get_subset(dataset, n, allowed=None):
    print(dataset, n, allowed)
    indices = []
    for i, d in enumerate(dataset):
        if len(indices) > n:
            break
        if allowed is None or d[1] in allowed:
            indices.append(i)
    return Subset(dataset, indices)

mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
weak_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
strong_transform = copy.deepcopy(weak_transform)
strong_transform.transforms.insert(1, RandomAugment(3, 5))

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)
    
datasets = {
    "train": datasets.CIFAR100(
            root="./data", train=True, transform=test_transform, download=True
        ),
    "test": datasets.CIFAR100(
                root="./data",
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )
}

#allowed_digits = {"all": list(range(10))}
for nr_examples in [8, 16, 32, 64, 128, 256]:
    net = resnet18(wnum_class=100)
    subset = get_subset(datasets["train"], nr_examples)
    dataloader = DataLoader(subset, 4, shuffle=True)
    dataloader_test = DataLoader(datasets["test"], 4)
    optimizer = Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = CrossEntropyLoss()

    cumulative_loss = 0
    i = 0

    for _ in range(4):
        for epoch in range(10):
            for batch in dataloader:
                i += 1
                data, labels = batch

                optimizer.zero_grad()

                data = Variable(data)
                out = net(data)

                loss = criterion(out, labels)
                cumulative_loss += float(loss)
                loss.backward()
                optimizer.step()
                print(i)
                if i % 50 == 0:
                    print("Loss: ", cumulative_loss / 100.0)
                    cumulative_loss = 0
        print("Accuracy", get_accuracy(net, dataloader_test))
    accuracies[nr_examples] = get_accuracy(net, dataloader_test)
    torch.save(net.state_dict(), "{}.pth".format(nr_examples))

with open("accuracies.txt", "w") as f:
    for k in accuracies:
        f.write("{}\t{}\n".format(k, accuracies[k]))

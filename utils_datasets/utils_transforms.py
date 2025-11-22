import torchvision.transforms as transforms
from .randaugment import RandomAugment
import copy


def get_dataset_transforms(args):
    if args.dataset == "hwf":
        weak_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        strong_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    elif args.dataset == "mmax" or args.dataset == "msum":
        weak_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        strong_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    elif args.dataset == "smallest_parent" and args.num_class == 10:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
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

    elif args.dataset == "smallest_parent" and args.num_class == 100:
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

    return weak_transform, strong_transform, test_transform

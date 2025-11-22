import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of ILP and CAROT")

# Dataset-related parameters
parser.add_argument(
    "--dataset",
    default="msum",
    type=str,
    choices=[
        "cifar10",
        "cifar100",
        "mmax",
        "msum",
        "hwf",
        "smallest_parent",
    ],
    help="dataset name (cifar10)",
)
parser.add_argument("--num-class", default=10, type=int, help="number of classes")
parser.add_argument(
    "--imb_type",
    default="exp",
    choices=["exp", "expr", "step", "original"],
    help="imbalance data type",
)
parser.add_argument(
    "--imb_ratio",
    default=5,
    type=float,
    help="imbalance ratio for long-tailed dataset generation",
)
parser.add_argument(
    "--imb_test",
    action="store_true", 
    default=False,
    help="use imbalanced test set",
)


parser.add_argument(
    "--data-dir",
    default="/Users/efi.tsamoura/Documents/data/pre-processed-data",
    type=str,
    help="experiment directory for loading pre-generated data",
)

parser.add_argument(
    "-b",
    "--batch-size",
    default=512,
    type=int,
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)

parser.add_argument(
    "--seed", default=1, type=int, help="seed for initializing training. "
)

parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    choices=["resnet18"],
    help="network architecture",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    help="number of data loading workers (default: 32)",
)
parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
parser.add_argument(
    "--epochs", default=1000, type=int, help="number of total epochs to run"
)

# Optimizer-related arguments
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "-lr_decay_epochs",
    type=str,
    default="700,800,900",
    help="where to decay lr, can be a list",
)
parser.add_argument(
    "-lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
)
parser.add_argument(
    "--cosine", action="store_true", default=False, help="use cosine lr schedule"
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-3,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-5)",
    dest="weight_decay",
)

# Statistics
parser.add_argument(
    "-p", "--print-freq", default=100, type=int, help="print frequency (default: 100)"
)

parser.add_argument("--save_ckpt", action="store_true", help="whether save the model")

parser.add_argument(
    "--exp-dir",
    default="experiment/CIFAR-10",
    type=str,
    help="experiment directory for saving checkpoints and logs",
)

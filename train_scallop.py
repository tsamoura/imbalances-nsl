import os
import random
import torch
from tqdm import tqdm

import torch.nn.functional as F
from networks.mnist import MNIST_NET
from networks.symbols import SymbolNet
from networks.resnet import resnet18
import torch.backends.cudnn as cudnn
import numpy as np

from arguments import parser
from utils_datasets.loader_arithmetic import ArithemticLoader
from utils_datasets.loader_hwf import HWFLoader
from utils.train_utils import (
    test_unbiased,
    save_checkpoint,
    mipll_estimate_empirical_distribution,
    test_records,
)
from utils.utils_statistics import AccurracyShot, adjust_learning_rate
from utils_scallop.arithmetic_solver import ArithmeticSolver
from utils_scallop.hwf_solver import HWFSolver
from utils_scallop.smallest_parent_solver import SmallestParentSolver
from utils_datasets.loader_smallest_parent import SmallestParentLoader
from utils_scallop.losses import bce_loss, nll_loss
from utils.utils_algorithms import ilp_pywraplp_mipll 
from utils_datasets.utils_margin import *

# Multi-Instance PLL-related arguments
parser.add_argument(
    "--size_partial_dataset",
    default=1000,
    type=int,
    help="number of partial training samples",
)

parser.add_argument(
    "--M",
    default=6,
    type=int,
    help="number of input instances per training sample",
)

parser.add_argument(
    "--ilp-training",
    action="store_true",
    default=True,
    help="use ilp regularization",
)

parser.add_argument(
    "--records", action="store_true", default=False, help="use records regularization"
)
parser.add_argument("--loss-fn", type=str, default="bce")
parser.add_argument("--top-k", type=int, default=1)
parser.add_argument("--jit", action="store_true")
parser.add_argument("--dispatch", type=str, default="parallel")
parser.add_argument(
    "--model",
    type=str,
    default=None,
)


# ILP-related parameters
parser.add_argument(
    "--epsilon_ilp",
    default=0.99,
    type=float,
    help="high-confidence selection threshold",
)
parser.add_argument(
    "--ilp_solver",
    default="pywrap",
    type=str,
    choices=["pywrap"],
    help="ilp solver (pywrap)",
)

parser.add_argument(
    "--continuous-relaxation",
    action="store_true",
    default=True,
    help="returning continuous relaxations of the linear program",
)

# Parameters related to (empirical) distribution estimation

parser.add_argument(
    "--gamma", default="0.05,0.01", type=str, help="distribution refinery param"
)
parser.add_argument(
    "--est_epochs",
    default=20,
    type=int,
    help="epochs for pre-estimating the class prior",
)

parser.add_argument(
    "--estimated-label-ratio",
    default="naive",
    type=str,
    choices=["gold", "partial", "naive"],
    help="use estimated label ratio",
)

parser.add_argument(
    "--validation-set-size",
    default=0,
    type=int,
    help="size of validation data",
)

class Trainer:
    def __init__(
        self,
        args,
        index_to_network_class,
        index_to_output_class,
        solver,
        optimizer,
        train_loader,
        test_loader,
        est_loader,
        acc_shot,
        loss,
        domain_restrictions = None
    ):
        self.args = args
        self.index_to_network_class = index_to_network_class
        self.solver = solver
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.index_to_output_class = index_to_output_class
        self.best_loss = 10000000000
        self.loss = loss
        self.est_loader = est_loader
        self.acc_shot = acc_shot
        self.domain_restrictions = domain_restrictions

    def train(
        self,
        total_epochs,
        emp_dist=None,
        is_est_dist=False,
        gamma=0,
    ):
        if is_est_dist:
            tip = "------------- Stage: Pre-Estimation --------------"
        else:
            tip = "------------- Stage: Final Training --------------"

        print(tip)
        with open(os.path.join(self.args.exp_dir, "result.log"), "a+") as f:
            f.write(tip + "\n")

        best_acc = 0
        for epoch in range(1, total_epochs + 1):
            is_best = False
            adjust_learning_rate(self.args, optimizer, epoch)

            self.train_epoch(epoch, emp_dist)

            if self.args.ilp_training and self.args.estimated_label_ratio == "naive":
                # How to estimate the empirical distribution in multi-instance PLL scenarios.
                # We have the options: a naive way, the gold one and the gold label ratios.
                emp_dist_train = mipll_estimate_empirical_distribution(
                    network, self.est_loader, num_class=self.args.num_class
                )
                # estimating empirical class prior by counting prediction
                emp_dist = emp_dist_train * gamma + emp_dist * (1 - gamma)
                # moving-average updating class prior

            if self.args.records:
                acc_test, acc_many, acc_med, acc_few = test_records(
                    self.acc_shot,
                    self.solver.network,
                    self.test_loader,
                    self.solver.feat_mean,
                    reshape_view=(-1, 16 * 4 * 4),
                )
            else:
                acc_test, acc_many, acc_med, acc_few = test_unbiased(
                    self.acc_shot, self.solver.network, self.test_loader
                )

            with open(os.path.join(self.args.exp_dir, "result.log"), "a+") as f:
                f.write(
                    "Epoch {}: Acc {:.2f}, Best Acc {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr {:.5f})\n".format(
                        epoch,
                        acc_test,
                        best_acc,
                        acc_many,
                        acc_med,
                        acc_few,
                        self.optimizer.param_groups[0]["lr"],
                    )
                )

            if acc_test > best_acc:
                best_acc = acc_test
                is_best = True

            if self.args.save_ckpt:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": self.args.arch,
                        "state_dict": self.solver.network.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    is_best=is_best,
                    filename="{}/checkpoint.pth.tar".format(self.args.exp_dir),
                    best_file_name="{}/checkpoint_best.pth.tar".format(
                        self.args.exp_dir
                    ),
                )
            # save checkpoints
        return emp_dist

    def train_epoch(self, epoch, emp_dist):
        self.solver.train()
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        for xs, ps in iter:
            xs = [x.cuda() for x in xs]
            self.optimizer.zero_grad()
            logits, predictions, output_probabilities, proofs = self.solver(xs)

            if self.args.ilp_training:
                bs = xs[0].shape[0]
                proofs = [proofs[b][p] for b, p in zip(range(bs), ps)]
                if self.args.ilp_solver == "pywrap":
                    pseudo_labels_soft = ilp_pywraplp_mipll(
                        proofs,
                        self.index_to_network_class,
                        predictions,
                        emp_dist,
                        self.args.epsilon_ilp,
                        self.args.continuous_relaxation,
                        self.domain_restrictions
                    )
                loss = 0
                for j in range(self.args.M):
                    # New inputs: logits
                    # New outpus: CE loss only 
                    l = F.cross_entropy(logits[j], pseudo_labels_soft[j].detach())
                    loss = loss + l
            else:
                ss = ps.cuda()
                loss = self.loss(output_probabilities, ss)
            loss.backward()
            self.optimizer.step()
            iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.ilp_training and args.records:
        raise ValueError("We cannot use records in conjunction with ILP-based training")
    if args.records and not (args.dataset == "msum" or args.dataset == "mmax"):
        raise ValueError(
            "At the moment, we support records only for the mmax and msum datasets"
        )

    torch.cuda.set_device(args.gpu)    
    [args.gamma1, args.gamma2] = [float(item) for item in args.gamma.split(",")]
    iterations = args.lr_decay_epochs.split(",")
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    print(args)
    args.imb_factor = 1.0 / args.imb_ratio
    model_path = "{ds}_M_{m}_imb_{it}{imf}_imbtest_{imt}_partial_samples_{ps}_sd_{seed}_record_{rc}".format(
        ds=args.dataset,
        m=args.M,
        ep=args.epochs,
        it=args.imb_type,
        imf=args.imb_factor,
        imt=args.imb_test,
        seed=args.seed,
        ps=args.size_partial_dataset,
        rc=args.records,
    )
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    domain_restrictions = None
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    if args.loss_fn == "nll":
        loss_fn = nll_loss
    elif args.loss_fn == "bce":
        loss_fn = bce_loss
    else:
        raise Exception(f"Unknown loss function `{args.loss_fn}`")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    if args.num_class == 10:
        many_shot_num = 3
        low_shot_num = 3
    elif args.num_class == 13:
        many_shot_num = 4
        low_shot_num = 4
    elif args.num_class == 100:
        many_shot_num = 33
        low_shot_num = 33

    if args.dataset == "mmax" or args.dataset == "msum":
        network = MNIST_NET()
        solver_func = ArithmeticSolver
        loader = ArithemticLoader(
            args, estimate_margin=args.estimated_label_ratio == "partial", scallop=True
        )
        domain_restrictions = loader.getDomainRestrictions()
    elif args.dataset == "hwf":
        network = SymbolNet()
        solver_func = HWFSolver
        loader = HWFLoader(
            args, estimate_margin=args.estimated_label_ratio == "partial", scallop=True
        )
        domain_restrictions = loader.getDomainRestrictions()
    elif args.dataset == "smallest_parent":
        network = resnet18(num_class=args.num_class)
        solver_func = SmallestParentSolver
        loader = SmallestParentLoader(
            args, estimate_margin=args.estimated_label_ratio == "partial", scallop=True
        )
        domain_restrictions = loader.getDomainRestrictions()
    else:
        raise NotImplementedError(
            "You have chosen an unsupported dataset. Please check and try again."
        )

    if args.model is not None:
        network.load_state_dict(torch.load(args.model)["state_dict"])

    network = network.cuda(args.gpu)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

    (
        train_loader,
        test_loader,
        est_loader,
        _,
        init_label_dist,
        train_label_cnt,
        index_to_network_class,
        index_to_output_class,
        gold_label_ratio,
        partial_ratio,
        label_vec_to_partial,
    ) = loader.load()

    if args.ilp_training and args.estimated_label_ratio == "partial":
        if args.dataset == "mmax":
            emp_dist = (
                solver_mirror_descent(
                    transiton_max,
                    partial_ratio,
                    args.num_class,
                    args.M,
                    n_iter=10000,
                )
                .detach()
                .numpy()
            )
        elif args.dataset == "msum":
            emp_dist = (
                solver_mirror_descent(
                    transiton_sum,
                    partial_ratio,
                    args.num_class,
                    args.M,
                    n_iter=10000,
                )
                .detach()
                .numpy()
            )
        elif args.dataset == "smallest_parent":
            emp_dist = (
                solver_mirror_descent(
                    transiton_smallest_common_parent,
                    partial_ratio,
                    args.num_class,
                    label_vec_to_partial,
                    n_iter=5000,
                )
                .detach()
                .numpy()
            )
    elif args.ilp_training and args.estimated_label_ratio == "naive":
        emp_dist = init_label_dist.unsqueeze(dim=1)

    solver = solver_func(network, index_to_network_class, index_to_output_class, args)
    
    solver = solver.cuda(args.gpu)

    acc_shot = AccurracyShot(
        train_label_cnt, args.num_class, many_shot_num, low_shot_num
    )

    trainer = Trainer(
        args,
        index_to_network_class,
        index_to_output_class,
        solver,
        optimizer,
        train_loader,
        test_loader,
        est_loader,
        acc_shot,
        loss_fn,
        domain_restrictions
    )
    if not args.ilp_training:
        trainer.train(
            total_epochs=args.epochs,
        )

    elif args.estimated_label_ratio == "gold":
        trainer.train(
            total_epochs=args.epochs,
            emp_dist=gold_label_ratio,
            gamma=args.gamma2,
        )

    elif args.estimated_label_ratio == "partial":
        trainer.train(
            total_epochs=args.epochs,
            emp_dist=emp_dist,
            gamma=args.gamma2,
        )
    elif args.estimated_label_ratio == "naive":
        emp_dist = trainer.train(
            total_epochs=args.est_epochs,
            emp_dist=emp_dist,
            is_est_dist=True,
            gamma=args.gamma1,
        )

        # Initialize network, optimizer and solver
        if args.dataset == "mmax" or args.dataset == "msum":
            network = MNIST_NET()
        elif args.dataset == "hwf":
            network = SymbolNet()
        elif args.dataset == "smallest_parent":
            network = resnet18(num_class=args.num_class)
        if args.model is not None:
            network.load_state_dict(torch.load(args.model)["state_dict"])
        
        network = network.cuda(args.gpu)
        
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
        solver = solver_func(
            network, index_to_network_class, index_to_output_class, args
        )
        
        solver = solver.cuda(args.gpu)

        trainer = Trainer(
            args,
            index_to_network_class,
            index_to_output_class,
            solver,
            optimizer,
            train_loader,
            test_loader,
            est_loader,
            acc_shot,
            loss_fn,
        )

        trainer.train(
            total_epochs=args.epochs,
            emp_dist=emp_dist,
            gamma=args.gamma2,
        )

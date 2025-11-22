import os
import random
import time
import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from networks.resnet import resnet18
from resnet_records.resnet import ResNet_s
from smallest_parent.abduction import SmallestParentAbduction
from smallest_parent.abstract_translator import AbstractTranslator
from smallest_parent.abstract_abduction import AbstractAbduction
from utils_datasets.loader_smallest_parent import SmallestParentLoader
from utils.utils_algorithms import ilp_pywraplp_mipll 
from typing import Dict
from utils_datasets.utils_margin import *
from smallest_parent.abducibles import get_abducibles

from utils_losses.utils_loss_smallest_parent import wmc_loss
import torch.nn.functional as F
from utils.train_utils import (
    test_unbiased,
    save_checkpoint,
    mipll_estimate_empirical_distribution,
    test_records,
)
from utils.utils_statistics import (
    AverageMeter,
    ProgressMeter,
    AccurracyShot,
    adjust_learning_rate,
)

torch.set_printoptions(precision=2, sci_mode=False)

from arguments import parser

# Empirical distribution-related parameters
parser.add_argument(
    "--queue_length",
    default=1,
    type=int,
    help="the queue size is queue_length*batch_size",
)
parser.add_argument(
    "--gamma", default="0.05,0.01", type=str, help="distribution refinery param"
)
parser.add_argument(
    "--est_epochs",
    default=20,
    type=int,
    help="epochs for pre-estimating the class prior",
)

# ILP-related parameters

parser.add_argument(
    "--ilp-training",
    action="store_true",
    default=False,
    help="use ilp regularization",
)

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
    default=False,
    help="returning continuous relaxations of the linear program",
)

# Multi-Instance PLL-related arguments
parser.add_argument(
    "--size_partial_dataset",
    default=1000,
    type=int,
    help="number of partial training samples",
)

parser.add_argument(
    "--estimated-label-ratio",
    default="naive",
    type=str,
    choices=["gold", "partial", "naive"],
    help="use estimated label ratio",
)

parser.add_argument(
    "--records", action="store_true", default=False, help="use records regularization"
)

class Trainer:
    def __init__(
        self,
        args,
        translator: AbstractTranslator,
        index_to_network_class: Dict,
        abduction: AbstractAbduction,
        train_loader,
        test_loader,
        est_loader,
        acc_shot,
        domain_restrictions = None
    ):
        self.args = args
        self.translator = translator
        self.index_to_network_class = index_to_network_class
        self.abduction = abduction
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.est_loader = est_loader
        # set loss functions (with pseudo-targets maintained)
        self.acc_shot = acc_shot
        self.domain_restrictions = domain_restrictions

    def train(
        self,
        network: nn.Module,
        optimizer: Optimizer,
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

        self.wmc_loss = wmc_loss()

        if self.args.ilp_training:
            queues = None
            if self.args.queue_length > 0 and queues is None:
                queues = [
                    torch.zeros(self.args.queue_length, self.args.num_class).cuda()
                    for _ in range(args.M)
                ]
            proofs_queue = None
            if self.args.queue_length > 0 and proofs_queue is None:
                proofs_queue = [[]] * self.args.queue_length
            # initialize queue for Sinkhorn iteration

        best_acc = 0
        for epoch in range(total_epochs):
            is_best = False
            adjust_learning_rate(self.args, optimizer, epoch)

            self.train_epoch(network, queues, proofs_queue, emp_dist, optimizer, epoch)

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
                    network,
                    self.test_loader,
                    self.feat_mean,
                    reshape_view=(-1, 16 * 4 * 4),
                )
            else:
                acc_test, acc_many, acc_med, acc_few = test_unbiased(
                    self.acc_shot, network, self.test_loader
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
                        optimizer.param_groups[0]["lr"],
                    )
                )

            if acc_test > best_acc:
                best_acc = acc_test
                is_best = True

            if not is_est_dist and self.args.save_ckpt:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": self.args.arch,
                        "state_dict": network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=is_best,
                    filename="{}/checkpoint.pth.tar".format(self.args.exp_dir),
                    best_file_name="{}/checkpoint_best.pth.tar".format(
                        self.args.exp_dir
                    ),
                )
            # save checkpoints

        return emp_dist

    def train_epoch(self, network, queues, proofs_queue, emp_dist, optimizer, epoch):
        batch_time = AverageMeter("Time", ":1.2f")
        data_time = AverageMeter("Data", ":1.2f")
        loss_sink_log = AverageMeter("Loss@Sink", ":2.2f")
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, loss_sink_log],  
            prefix="Epoch: [{}]".format(epoch),
        )

        # switch to train mode
        network.train()

        end = time.time()

        bs = self.args.batch_size
        for i, (x1_w, _, x2_w, _, Y1, Y2, y1, y2, s1, s2, _, _) in enumerate(
            self.train_loader
        ):
            # measure data loading time
            data_time.update(time.time() - end)

            X1_w, Y1 = x1_w.cuda(), Y1.cuda()
            X2_w, Y2 = x2_w.cuda(), Y2.cuda()

            
            if not self.args.ilp_training and not self.args.records:
                logits_X1_w = network(X1_w)
                logits_X2_w = network(X2_w)
                pseudo_loss_vec = self.wmc_loss(
                    logits_X1_w, Y1, logits_X2_w, Y2, s1, s2
                )

                loss = pseudo_loss_vec.mean()
                loss_sink_log.update(pseudo_loss_vec.mean().item())
            if not self.args.ilp_training and self.args.records:
                logits_X1_w, feat1_w = network(X1_w)
                logits_X2_w, feat2_w = network(X2_w)
                pseudo_loss_vec = self.wmc_loss(
                    logits_X1_w,
                    Y1,
                    logits_X2_w,
                    Y2,
                    s1,
                    s2,
                    True,
                    network,
                    feat1_w,
                    feat2_w,
                )

                loss = pseudo_loss_vec.mean()
                loss_sink_log.update(pseudo_loss_vec.mean().item())
            else:
                logits_X1_w = network(X1_w)
                logits_X2_w = network(X2_w)
                logits_x = [logits_X1_w, logits_X2_w]
                sinkhorn_costs = [
                    F.softmax(logits.detach(), dim=1) for logits in logits_x
                ]
                # time to use queue, output now represent queue+output
                predictions = [
                    sinkhorn_cost.detach() for sinkhorn_cost in sinkhorn_costs
                ]
                proofs = list()
                for j in range(self.args.batch_size):
                    proofs.append(
                        self.abduction._solve(Y1[j], Y2[j], self.index_to_network_class)
                    )
                # time to use queue, output now represent queue+output
                # The proofs that are passed to the linear solvers should correspond to those of the
                # elements in the queue
                if queues is not None:
                    for j, queue in enumerate(queues):
                        if not torch.all(queue[-1, :] == 0):
                            predictions[j] = torch.cat((queue, predictions[j]))
                            if j == 0:
                                proofs = proofs_queue + proofs
                        # fill the queue
                        queue[bs:] = queue[:-bs].clone().detach()
                        queue[:bs] = predictions[j][-bs:].clone().detach()
                        if j == 0:
                            proofs_queue[bs:] = proofs_queue[:-bs].copy()
                            proofs_queue[:bs] = proofs[-bs:].copy()

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
                pseudo_labels = [
                    pseudo_label_soft[-bs:] for pseudo_label_soft in pseudo_labels_soft
                ]

                loss = 0
                for j in range(self.args.M):
                    loss += F.cross_entropy(logits_x[j], pseudo_labels[j].detach())
                loss /= self.args.M
                loss_sink_log.update(loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)


if __name__ == "__main__":
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    [args.gamma1, args.gamma2] = [float(item) for item in args.gamma.split(",")]
    iterations = args.lr_decay_epochs.split(",")
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))
    args.queue_length *= args.batch_size
    print(args)
    args.imb_factor = 1.0 / args.imb_ratio

    model_path = (
        "{ds}_M_{m}_imb_{it}{imf}_imbtest_{imt}_partial_samples_{ps}_sd_{seed}".format(
            ds=args.dataset,
            m=args.M,
            ep=args.epochs,
            it=args.imb_type,
            imf=args.imb_factor,
            imt=args.imb_test,
            seed=args.seed,
            ps=args.size_partial_dataset,
        )
    )
    args.exp_dir = os.path.join(args.exp_dir, model_path)
    domain_restrictions = None
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    abducibles, exclusive = get_abducibles(args.M, args.num_class)
    translator = AbstractTranslator(abducibles, exclusive)
    abduction = SmallestParentAbduction(translator, args.M, args.num_class)

    loader = SmallestParentLoader(
        args, estimate_margin=args.estimated_label_ratio == "partial", scallop=False
    )
    domain_restrictions = loader.getDomainRestrictions()
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

    if args.num_class == 10:
        many_shot_num = 3
        low_shot_num = 3
    elif args.num_class == 100:
        many_shot_num = 33
        low_shot_num = 33

    acc_shot = AccurracyShot(
        train_label_cnt, args.num_class, many_shot_num, low_shot_num
    )

    trainer = Trainer(
        args,
        translator,
        index_to_network_class,
        abduction,
        train_loader,
        test_loader,
        est_loader,
        acc_shot,
        domain_restrictions
    )

    if not args.ilp_training:
        if args.records:
            network = ResNet_s(name="resnet18", num_class=args.num_class)
        else:
            network = resnet18(num_class=args.num_class)
        network = network.cuda(args.gpu)
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
        trainer.train(
            total_epochs=args.epochs,
        )

    elif args.estimated_label_ratio == "gold":
        network = resnet18(num_class=args.num_class)
        network = network.cuda(args.gpu)
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        trainer.train(
            total_epochs=args.epochs,
            emp_dist=gold_label_ratio,
            gamma=args.gamma2,
        )

    elif args.estimated_label_ratio == "partial":
        network = resnet18(num_class=args.num_class)
        network = network.cuda(args.gpu)
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        trainer.train(
            network=network,
            optimizer=optimizer,
            total_epochs=args.epochs,
            emp_dist=emp_dist,
            gamma=args.gamma2,
        )
    elif args.estimated_label_ratio == "naive":
        network = resnet18(num_class=args.num_class)
        network = network.cuda(args.gpu)
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        emp_dist = trainer.train(
            network=network,
            optimizer=optimizer,
            total_epochs=args.est_epochs,
            emp_dist=emp_dist,
            is_est_dist=True,
            gamma=args.gamma1,
        )

        network = resnet18(num_class=args.num_class)
        network = network.cuda(args.gpu)
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        trainer.train(
            network=network,
            optimizer=optimizer,
            total_epochs=args.epochs,
            emp_dist=emp_dist,
            gamma=args.gamma2,
        )

# Testing-time imbalance mitigation using LA and CAROT for the MAX-M scenarios.
# The script support long-tail training/long-tail testing settings.

from utils.train_utils import test_unbiased, test_otla, test_partial, test_partial_otla
from argparse import Namespace
from utils_datasets.utils_margin import *
from utils.utils_statistics import AccurracyShot
from utils.utils_algorithms import robust_semi_sinkhorn
import random
import torch.backends.cudnn as cudnn
import numpy as np

from networks.mnist import MNIST_NET
from utils_datasets.loader_arithmetic import ArithemticLoader


torch.set_printoptions(linewidth=100)


def carot_experiment(
    model_path,
    imb_type,
    imb_ratio,
    M,
    size_partial_dataset,
    seed,
    dataset="mmax",
    validation_set_size=256,
    validation_epoch=5,
    use_gold_label=False,
):
    if dataset == "mmax":
        func = max
        transition = transiton_max
    elif dataset == "msum":
        func = sum
        transition = transiton_sum
    else:
        func = None
        transition = None

    separator = "-" * 66
    print(
        separator
        + "\n[Testing imbalance type {it} with ratio {ib}, M={m}, dataset size={ds}]\n".format(
            it=imb_type,
            ib=imb_ratio,
            m=M,
            ds=size_partial_dataset,
        )
        + separator
    )

    args = Namespace(
        dataset=dataset,
        num_class=10,
        batch_size=32,
        imb_type=imb_type,
        imb_ratio=imb_ratio,
        imb_test=True,
        seed=seed,
        M=M,
        validation_set_size=validation_set_size,
        size_partial_dataset=size_partial_dataset,
        data_dir="data/pre-processed-data",
    )
    args.imb_factor = 1.0 / args.imb_ratio

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    # imbalanced test set
    print("-----------Testing perfomance at imbalanced testing set-----------")

    loader = ArithemticLoader(args, estimate_margin=True, scallop=True)

    (
        _,
        test_loader,
        _,
        val_loader,
        _,
        train_label_cnt,
        _,
        _,
        gold_label_ratio,
        partial_ratio,
        _,
    ) = loader.load()

    print("Using a validation set of size:", len(val_loader.dataset))
    acc_shot = AccurracyShot(train_label_cnt, args.num_class, 3, 3)
    if use_gold_label:
        est_ratio = gold_label_ratio
    else:
        est_ratio = solver_mirror_descent(
            transition, partial_ratio, 10, args.M, n_iter=10000
        ).detach()
    print("Estimated ratio: {}".format(est_ratio))
    print("Partial ratio: {}".format(partial_ratio))
    model = MNIST_NET().cuda()
    model.load_state_dict(torch.load(model_path)["state_dict"])

    # vanilla
    acc_vanilla, _, _, _ = test_unbiased(acc_shot, model, test_loader, verbose=True)
    print("Vanilla model acc: {}".format(acc_vanilla))

    # LA
    best_acc, best_tau = 0, 0
    for tau in torch.logspace(-5, 4, 20):
        acc_la = test_partial(model, val_loader, func, est_ratio, tau)
        if acc_la > best_acc:
            best_acc, best_tau = acc_la, tau
    acc_la, _, _, _ = test_unbiased(
        acc_shot, model, test_loader, verbose=True, ratio=est_ratio, tau=best_tau
    )
    print("LA acc: {}, tau: {}".format(acc_la, best_tau))

    # CAROT
    best_acc, best_lamd, best_tau = 0, 1, 1
    for lamd in torch.logspace(-5, 4, 10):
        for tau in torch.logspace(-5, 4, 20):
            acc_otla = test_partial_otla(
                model,
                val_loader,
                func,
                robust_semi_sinkhorn.apply,
                est_ratio,
                lamd,
                tau,
                epochs=validation_epoch,
            )
            if acc_otla > best_acc:
                best_acc, best_lamd, best_tau = acc_otla, lamd, tau

    acc_otla, _, _, _ = test_otla(
        acc_shot,
        model,
        test_loader,
        robust_semi_sinkhorn.apply,
        est_ratio,
        best_lamd,
        best_tau,
        apply_softmax=False,
        verbose=True,
    )
    print("CAROT: acc {}, lamd {}, tau {}".format(acc_otla, best_lamd, best_tau))

    return acc_vanilla, acc_la, acc_otla


model_path = "path to model to test"

acc_vanilla, acc_la, acc_otla = carot_experiment(
    model_path=model_path,
    imb_type="expr",
    imb_ratio=5,
    M=3, # M is set as in our paper
    size_partial_dataset=3000, # number of partial training samples
    seed=2,
    dataset="mmax", # "msum" 
    validation_set_size=256,
    validation_epoch=1,
)

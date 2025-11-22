import sys
sys.path.append("../")
from utils.train_utils import (
    test_unbiased,
    test_otla,
    test_partial_cifar,
    test_partial_cifar_otla,
    test_partial,
    test_partial_otla,
)
from utils_datasets.loader_hwf import HWFLoader
from utils_datasets.loader_smallest_parent import SmallestParentLoader
from utils_datasets.loader_arithmetic import ArithemticLoader
from argparse import Namespace
from utils_datasets.utils_margin import *
from utils.utils_statistics import AccurracyShot
from utils.utils_algorithms import robust_semi_sinkhorn
from networks.mnist import MNIST_NET
from networks.resnet import resnet18
from resnet_records.resnet import ResNet_s
from networks.symbols import SymbolNet
import random
import torch.backends.cudnn as cudnn
import numpy as np

torch.set_printoptions(linewidth=100)

def test_hwf(
    model_path,
    imb_type,
    imb_ratio,
    M,
    size_partial_dataset,
    seed,
    imb_test=True,
    dataset="hwf",
):
    separator = "-" * 66
    print(
        separator
        + "\n[Testing imbalance type {it} with ratio {ib} M = {M}, size={ds}, seed = {sd}]\n".format(
            it=imb_type, ib=imb_ratio, M=M, ds=size_partial_dataset, sd=seed,
        )
        + separator
    )

    args = Namespace(
        dataset=dataset,
        num_class=13,
        batch_size=256,
        imb_type=imb_type,
        imb_ratio=imb_ratio,
        imb_test=imb_test,
        seed=seed,
        size_partial_dataset=size_partial_dataset,
        data_dir="data/pre-processed-data",
        M=M
    )
    args.imb_factor = 1.0 / args.imb_ratio

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    print("------------Testing perfomance at balanced testing set------------")
    loader = HWFLoader(
        args=args
    )
    (
        _,
        test_loader,
        _,
        _,
        _,
        train_label_cnt,
        _,
        _,
        _,
        _,
        _,
    ) = loader.load()


    acc_shot = AccurracyShot(train_label_cnt, args.num_class, 3, 3)

    # test vanilla
    model = SymbolNet().cuda()
    model.load_state_dict(torch.load(model_path)["state_dict"])

    acc_vanilla, _, _, _ = test_unbiased(acc_shot, model, test_loader, verbose=False)
    print("vanilla Model acc: {}".format(acc_vanilla))
    return acc_vanilla
    

def la_carot_smallest_parent(
    model_path,
    imb_type,
    imb_ratio,
    size_partial_dataset,
    seed,
    imb_test=True,
    dataset="smallest_parent",
    validation_set_size=256,
    validation_epoch=5,
    use_gold_label=False,
    records=False,
):
    separator = "-" * 66
    print(
        separator
        + "\n[Testing imbalance type {it} with ratio {ib}, size={ds}, seed = {sd}]\n".format(
            it=imb_type, ib=imb_ratio, ds=size_partial_dataset, sd=seed
        )
        + separator
    )

    args = Namespace(
        dataset=dataset,
        num_class=10,
        batch_size=256,
        imb_type=imb_type,
        imb_ratio=imb_ratio,
        imb_test=imb_test,
        seed=seed,
        size_partial_dataset=size_partial_dataset,
        data_dir="data/pre-processed-data",
        M=2
    )
    args.imb_factor = 1.0 / args.imb_ratio

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    print("------------Testing perfomance at balanced testing set------------")
    loader = SmallestParentLoader(
        args=args, estimate_margin=True, validation_set_size=validation_set_size
    )
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
        partial_label_ratio,
        label_vec_to_partial,
    ) = loader.load()

    if use_gold_label:
        est_ratio = gold_label_ratio
    else:
        est_ratio = solver_mirror_descent(
            transiton_smallest_common_parent,
            partial_label_ratio,
            10,
            label_vec_to_partial,
            n_iter=5000,
        ).detach()
    print("Estimated ratio: {}".format(est_ratio))

    acc_shot = AccurracyShot(train_label_cnt, args.num_class, 3, 3)

    # test vanilla
    if records:
        model = ResNet_s(name="resnet18", num_class=args.num_class).cuda()
    else:
        model = resnet18(num_class=args.num_class).cuda()
    model.load_state_dict(torch.load(model_path)["state_dict"])

    acc_vanilla, _, _, _ = test_unbiased(acc_shot, model, test_loader, verbose=False)
    print("vanilla Model acc: {}".format(acc_vanilla))

    # LA
    best_acc, best_tau = 0, 0
    for tau in torch.logspace(-5, 4, 20):
        acc_la = test_partial_cifar(
            model, val_loader, label_vec_to_partial, est_ratio, tau
        )
        if acc_la > best_acc:
            best_acc, best_tau = acc_la, tau
    acc_la, _, _, _ = test_unbiased(
        acc_shot, model, test_loader, verbose=True, ratio=est_ratio, tau=best_tau
    )
    print("LA acc: {}, tau: {}".format(acc_la, best_tau))

    # test carot
    best_acc, best_lamd, best_tau = 0, 0, 0
    for lamd in torch.logspace(-5, 4, 10):
        for tau in torch.logspace(-5, 4, 20):
            acc_otla = test_partial_cifar_otla(
                model,
                val_loader,
                label_vec_to_partial,
                robust_semi_sinkhorn.apply,
                est_ratio,
                lamd,
                tau,
                epochs=validation_epoch,
            )
            if acc_otla > best_acc:
                best_acc, best_lamd, best_tau = acc_otla, lamd, tau

    acc_carot, _, _, _ = test_otla(
        acc_shot,
        model,
        test_loader,
        robust_semi_sinkhorn.apply,
        est_ratio,
        best_lamd,
        best_tau,
        verbose=True,
    )
    print("CAROT: acc {}, lamd {}, tau {}".format(acc_carot, best_lamd, best_tau))

    return acc_vanilla, acc_la, acc_carot

def la_carot_arithmetic(
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
    if imb_type != "original":
        args = Namespace(
            dataset=dataset,
            num_class=10,
            batch_size=32,
            imb_type=imb_type,
            imb_ratio=imb_ratio,
            imb_test=True,
            seed=seed,
            M=M,
            size_partial_dataset=size_partial_dataset,
            data_dir="data/pre-processed-data",
        )
        
    else:
        args.imb_ratio = 1
        args = Namespace(
            dataset=dataset,
            num_class=10,
            batch_size=256,
            imb_type="original",
            imb_ratio=1,
            imb_test=False,
            seed=seed,
            M=M,
            size_partial_dataset=size_partial_dataset,
            data_dir="data/pre-processed-data",
        )

    print(
        separator
        + "\n[Testing imbalance type {it} with ratio {ib}, M={m}, dataset size={ds}]\n".format(
            it=imb_type,
            ib=args.imb_ratio,
            m=M,
            ds=size_partial_dataset,
        )
        + separator
    )

    args.imb_factor = 1.0 / args.imb_ratio

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    loader = ArithemticLoader(
        args, estimate_margin=True, validation_set_size=validation_set_size
    )
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
        partial_label_ratio,
        _,
    ) = loader.load()

    print("Using a validation set of size:", len(val_loader.dataset))
    acc_shot = AccurracyShot(train_label_cnt, args.num_class, 3, 3)
    if use_gold_label:
        est_ratio = gold_label_ratio
    else:
        est_ratio = solver_mirror_descent(
            transition, partial_label_ratio, 10, args.M, n_iter=10000
        ).detach()
    print("Estimated ratio: {}".format(est_ratio))
    print("Partial ratio: {}".format(partial_label_ratio))
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
    if imb_type != "original":
        best_acc, best_lamd, best_tau = 0, 1, 1
    else:
        best_acc, best_lamd, best_tau = 0, 0, 0
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
        verbose=True,
    )
    print("CAROT: acc {}, lamd {}, tau {}".format(acc_otla, best_lamd, best_tau))

    return acc_vanilla, acc_la, acc_otla

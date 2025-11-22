# Testing-time imbalance mitigation using LA and CAROT for the MAX-M scenarios.
# The script support both long-tail training/long-tail testing and imbalance-free training/imbalance-free testing settings.

# Below, we provide an example for long-tail training/long-tail testing. For imbalance-free training/imbalance-free testing, please only follow the instructions below.

from utils.train_utils import test_unbiased, test_otla, test_partial_cifar_otla
from argparse import Namespace
from utils_datasets.utils_margin import *
from utils.utils_statistics import AccurracyShot
from utils.utils_algorithms import robust_semi_sinkhorn
from utils_datasets.loader_smallest_parent import SmallestParentLoader
from networks.resnet import resnet18
from resnet_records.resnet import ResNet_s
import random
import torch.backends.cudnn as cudnn
import numpy as np

torch.set_printoptions(linewidth=100)


from utils.train_utils import (
    test_unbiased,
    test_otla,
    test_partial_cifar,
    test_partial_cifar_otla,
)

from argparse import Namespace
from utils_datasets.utils_margin import *
from utils.utils_statistics import AccurracyShot
from utils.utils_algorithms import robust_semi_sinkhorn

import random
import torch.backends.cudnn as cudnn
import numpy as np

torch.set_printoptions(linewidth=100)


def carot_experiment(
    model_path,
    imb_type,
    imb_ratio,
    imb_test=True,
    size_partial_dataset=10000,
    seed=1,
    validation_set_size=256,
    validation_epoch=5,
    use_gold_label=False,
    apply_softmax=True,
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
        num_class=10,
        batch_size=256,
        imb_type=imb_type,
        imb_ratio=imb_ratio,
        imb_test=imb_test,
        seed=seed,
        validation_set_size=validation_set_size,
        size_partial_dataset=size_partial_dataset,
        data_dir="data/pre-processed-data",
    )
    args.imb_factor = 1.0 / args.imb_ratio

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True

    print("------------Testing perfomance at balanced testing set------------")

    loader = SmallestParentLoader(args, estimate_margin=True, scallop=False)
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
        label_vec_to_partial,
    ) = loader.load()

    if use_gold_label:
        est_ratio = gold_label_ratio
    else:
        est_ratio = solver_mirror_descent(
            transiton_smallest_common_parent,
            partial_ratio,
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
        apply_softmax=apply_softmax,
        verbose=True,
    )
    print("CAROT: acc {}, lamd {}, tau {}".format(acc_carot, best_lamd, best_tau))

    return acc_vanilla, acc_la, acc_carot


model_path = "path to model to test"

acc_vanilla, acc_la, acc_carot = carot_experiment(
    model_path=model_path,
    imb_type="exp",  # change to "original" for imbalance-free training and testing
    imb_ratio=5,  # change to 1 for imbalance-free training and testing
    imb_test=True,  # change to False for imbalance-free training and testing
    size_partial_dataset=10000, # number of partial training samples
    seed=1,
    validation_set_size=1024,
)

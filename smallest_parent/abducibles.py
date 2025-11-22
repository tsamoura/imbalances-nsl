import sys

sys.path.append("../")
from utils_datasets.cifar_metadata import cifar10_classes, cifar100_classes


def get_abducibles(M, num_class):
    abducibles = []
    if num_class == 10:
        base_classes = cifar10_classes
    elif num_class == 100:
        base_classes = cifar100_classes
    for i in range(M):
        for j in base_classes:
            abducibles.append("at({},{})".format(j, i))

    exclusive = []
    for i in range(M):
        me = list()
        for j in base_classes:
            me.append("at({},{})".format(j, i))
        exclusive.append(me)
    return abducibles, exclusive
import sys

sys.path.append("../")
from .abstract_abduction import AbstractAbduction
from .abstract_translator import AbstractTranslator

import itertools
import torch
import numpy as np

class SmallestParentAbduction(AbstractAbduction):
    def __init__(self, translator: AbstractTranslator, M, num_class):
        AbstractAbduction.__init__(self, translator)
        assert M == 2
        self.M = M

    def _solve(self, lineage1, lineage2, index_to_network_class):
        proofs = list()
        if (
            np.array_equal(lineage1.cpu().numpy(), lineage2.cpu().numpy())
            and torch.sum(lineage1) == 1
        ):
            indices = torch.nonzero(lineage1, as_tuple=False)[0].tolist()
            position = indices[0]
            proof = [
                "at({},{})".format(index_to_network_class[position], 0),
                "at({},{})".format(index_to_network_class[position], 1),
            ]
            proofs.append(proof)
        elif (
            np.array_equal(lineage1.cpu().numpy(), lineage2.cpu().numpy())
            and torch.sum(lineage1) == 2
        ):
            # In CIFAR10, this can only happen, when the base classes have exactly the same parent.
            # For instance:
            # home_land_animals <= cat, dog
            # wild_land_animals <= deer, horse
            # other_animals <= bird, frog
            # other_transportation <= airplane, ship
            # If there are only two base classes, e.g. as in other_animals <= bird, frog and other_transportation <= airplane, ship,
            # then, do the trick from below, to compute the WMC
            indices = torch.nonzero(lineage1, as_tuple=True)[0].tolist()
            position1 = indices[0]
            position2 = indices[1]
            proof1 = [
                "at({},{})".format(index_to_network_class[position1], 0),
                "at({},{})".format(index_to_network_class[position2], 1),
            ]
            proof2 = [
                "at({},{})".format(index_to_network_class[position2], 0),
                "at({},{})".format(index_to_network_class[position1], 1),
            ]
            proofs.append(proof1)
            proofs.append(proof2)

        elif np.sum(np.multiply(lineage1.cpu().numpy(), lineage2.cpu().numpy())) > 0:
            raise ValueError(
                "This case is not supported due to assumptions on the benchmark, i.e., that there should be two preconditions per implication."
            )
        else:
            indices1 = torch.nonzero(lineage1, as_tuple=True)[0].tolist()
            indices2 = torch.nonzero(lineage2, as_tuple=True)[0].tolist()
            for p in itertools.product(indices1, indices2):
                proof = [
                    "at({},{})".format(index_to_network_class[p[0]], 0),
                    "at({},{})".format(index_to_network_class[p[1]], 1),
                ]
                proofs.append(proof)
            for p in itertools.product(indices2, indices1):
                proof = [
                    "at({},{})".format(index_to_network_class[p[0]], 0),
                    "at({},{})".format(index_to_network_class[p[1]], 1),
                ]
                proofs.append(proof)

        return proofs
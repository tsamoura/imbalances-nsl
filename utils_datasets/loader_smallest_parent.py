import numpy as np
import torch
import sys
import os
import networkx
from .loader import DatasetLoader
import torchvision.datasets as datasets
from utils_datasets.mipll_datasets import Scallop_Dataset, MIPLL_Dataset, SMALLEST_PARENT_Dataset

sys.path.append("../")
from utils_datasets.utils_data import (
    gen_imbalanced_data,
)

from utils_datasets.cifar_metadata import (
    cifar10_theory,
    cifar100_theory,
    find_base_classes,
    create_is_a_graph,
    get_all_ancestors,
    cifar10_classes,
    cifar100_classes,
)
from utils_datasets.cifar_metadata import (
    cifar10_output_classes,
    cifar100_output_classes,
)
from collections import Counter

def get_smallest_common_ancestor(
    object_type1s, object_type2s, is_a_graph, base_classes
):
    s1 = []
    s2 = []
    p = []
    for object_type1, object_type2 in zip(object_type1s, object_type2s):
        superclasses1 = get_all_ancestors(object_type1, is_a_graph)
        superclasses2 = get_all_ancestors(object_type2, is_a_graph)
        assert len(superclasses1) > 0
        assert len(superclasses2) > 0

        # If they are of the same type, then this their common ancestor
        if object_type1 == object_type2:
            s1.append(object_type1)
            s2.append(object_type2)
            p.append(object_type2)
        # If the base objects have the same parent.
        # In CIFAR10, this can only happen in the following cases:
        # other_transportation <= airplane, ship
        # other_animals <= bird, frog
        # home_land_animals <= cat, dog
        # wild_land_animals <= deer, horse
        elif (
            superclasses1[0] == superclasses2[0]
            and len(find_base_classes(superclasses1[0], is_a_graph, base_classes)) <= 2
        ):
            s1.append(superclasses1[0])
            s2.append(superclasses2[0])
            p.append(superclasses1[0])
        elif (
            superclasses1[0] == superclasses2[0]
            and len(find_base_classes(superclasses1[0], is_a_graph, base_classes)) > 2
        ):
            s1.append(-1)
            s2.append(-1)
            p.append(-1)
        else:
            no_common_parent = True
            for sc1 in superclasses1:
                for sc2 in superclasses2:
                    if sc1 == sc2 and no_common_parent:
                        # Example: animals <= mammals, non_mammals
                        # Then, the first image should be a mammal and the second should be a non_mammal or vice versa
                        # Find the two children of sc1
                        ingoing_edges = list(is_a_graph.in_edges(sc1))
                        assert len(ingoing_edges) == 2
                        src1 = ingoing_edges[0][0]
                        src2 = ingoing_edges[1][0]
                        s1.append(src1)
                        s2.append(src2)
                        p.append(sc1)
                        no_common_parent = False
            # If two pairs of labels have no common ancestor, then we return partial label None
            # The None partial label will get all the possible classes as candidates ones
            if no_common_parent:
                raise RuntimeError("There should be a common ancestor")

    return s1, s2, p


class SmallestParentLoader(DatasetLoader):
    def __init__(
        self, args, estimate_margin=False, validation_set_size=0, scallop=False
    ):
        DatasetLoader.__init__(
            self, args, estimate_margin, validation_set_size, scallop
        )

    def getDomainRestrictions(self):
        return None

    def load(self):
        index_to_output_class = self.getIndexToOutputClass()
        output_class_to_index = {c: i for i, c in index_to_output_class.items()}
        
        (
            x1,
            y1,
            Y1,
            s1,
            x2,
            y2,
            Y2,
            s2,
            p,
            index_to_network_class,
            gold_label_ratio,
            partial_ratio,
            label_vec_to_partial,
        ) = self.get_pll_dataset()

        init_label_dist = torch.ones(self.args.num_class) / self.args.num_class
        # Be careful. If you choose not many partial training samples, then it is likely that not all classes are sampled,
        # especially, when specifying argument 'exp'.
        # Hence, this line of code might return an exception.
        train_label_cnt = self.get_train_label_cnt([y1,y2])
        # train_label_cnt is also used for intialize Acc-shot object
        
        if self.scallop:
            train_pll_dataset = Scallop_Dataset(
                [x1, x2],
                [y1.float(), y2.float()],
                [output_class_to_index[element] for element in p],
                self.args.size_partial_dataset,
                self.weak_transform,
            )

            train_pll_dataset_loader = torch.utils.data.DataLoader(
                dataset=train_pll_dataset,
                collate_fn=Scallop_Dataset.collate_fn,
                batch_size=self.args.batch_size,
                shuffle=True,
            )

        else:
            train_pll_dataset = SMALLEST_PARENT_Dataset(
                x1,
                x2,
                Y1.float(),
                Y2.float(),
                y1.float(),
                y2.float(),
                s1,
                s2,
                p,
                self.weak_transform,
                self.strong_transform,
            )

            train_pll_dataset_loader = torch.utils.data.DataLoader(
                dataset=train_pll_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )

        # test loader
        test_dataset = self.createTestDataset()
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.args.batch_size * 4,
            shuffle=True,
            num_workers=4,
        )
        # set test dataloader

        est_dataset = MIPLL_Dataset(
            [x1, x2],
            [y1.float(), y2.float()],
            p,
            self.args.size_partial_dataset,
            self.weak_transform,
        )

        est_loader = torch.utils.data.DataLoader(
            dataset=est_dataset,
            batch_size=self.args.batch_size * 4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # validation loader
        val_loader = None
        if self.validation_set_size > 0:
            x1, y1, Y1, s1, x2, y2, Y2, s2, p, index_to_network_class = self.get_pll_dataset(validation=True)
            val_dataset = SMALLEST_PARENT_Dataset(
                x1,
                x2,
                Y1.float(),
                Y2.float(),
                y1.float(),
                y2.float(),
                s1,
                s2,
                p,
                self.weak_transform,
                self.strong_transform,
            )

            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

        return (
            train_pll_dataset_loader,
            test_loader,
            est_loader,
            val_loader,
            init_label_dist,
            train_label_cnt,
            index_to_network_class,
            index_to_output_class,
            gold_label_ratio,
            partial_ratio,
            label_vec_to_partial,
        )
    
    def getIndexToOutputClass(self):
        if self.args.num_class == 10:
            index_to_output_class = {i: c for i, c in enumerate(cifar10_output_classes)}
        elif self.args.num_class == 100:
            index_to_output_class = {
                i: c for i, c in enumerate(cifar100_output_classes)
            }
        return index_to_output_class

    def get_pll_dataset(self, validation=False):
        if validation:
            size = self.validation_set_size
        else:
            size = self.args.size_partial_dataset
        print("==> Loading local data copy in the long-tailed setup")
        data_file = self.getDataFileName(validation, size)
        save_path = os.path.join(self.args.data_dir, data_file)
        if not os.path.exists(save_path):
            # First, the long-tailed learning is known to be unstable,
            # we recommend running SoLar with a pre-processed data copy,
            # which can be used for other baseline models as well.

            is_a_graph = networkx.DiGraph()
            data, labels, classes, class_to_idx = self.loadOriginalTrainDataset()
            if self.args.num_class == 10:
                # Load the hierachies for CIFAR10 and populate the graph with edges from subclasses to superclasses
                is_a_graph = create_is_a_graph(cifar10_theory)
                base_classes = cifar10_classes
            elif self.args.num_class == 100:
                # Load the hierachies for CIFAR100 and populate the graph with edges from subclasses to superclasses
                is_a_graph = create_is_a_graph(cifar100_theory)
                base_classes = cifar100_classes

            index_to_network_class = {idx: cls for cls, idx in class_to_idx.items()}
            train_data, train_labels = gen_imbalanced_data(
                data,
                labels,
                self.args.num_class,
                self.args.imb_type,
                self.args.imb_factor,
            )

            # Sample args.size_partial_dataset images (x1)
            random1 = np.random.choice(range(0, len(train_data) - 1), size)
            # Sample args.size_partial_dataset images (x2)
            random2 = np.random.choice(range(0, len(train_data) - 1), size)

            # Get the x1 images
            x1 = train_data[random1]
            # Get the x2 images
            x2 = train_data[random2]
            # Get the gold labels for x1 images
            y1 = train_labels[random1]
            # Get the gold labels for x2 images
            y2 = train_labels[random2]

            # Get classes for x1 images
            c1 = [index_to_network_class[i] for i in y1.numpy()]
            # Get classes for x2 images
            c2 = [index_to_network_class[i] for i in y2.numpy()]

            s1, s2, pi = get_smallest_common_ancestor(c1, c2, is_a_graph, base_classes)
            # Find all indices where s1 and s2 are -1
            i1 = [i for i, x in enumerate(s1) if x != -1]
            i2 = [i for i, x in enumerate(s2) if x != -1]

            # Compute the partial vectors based on superclasses for (x1)
            Y1_seeds = [
                find_base_classes(i, is_a_graph, classes) if i != -1 else list()
                for i in s1
            ]
            # Compute the partial vectors based on superclasses for (x2)
            Y2_seeds = [
                find_base_classes(i, is_a_graph, classes) if i != -1 else list()
                for i in s2
            ]

            # Create PLL vectors
            Y1 = np.zeros((size, self.args.num_class))
            for r, seeds in enumerate(Y1_seeds):
                for s in seeds:
                    Y1[r, class_to_idx[s]] = 1

            Y2 = np.zeros((size, self.args.num_class))
            for r, seeds in enumerate(Y2_seeds):
                for s in seeds:
                    Y2[r, class_to_idx[s]] = 1

            x1 = x1[i1, :, :, :]
            x2 = x2[i2, :, :, :]
            y1 = y1[i1]
            y2 = y2[i2]
            Y1 = Y1[i1]
            Y2 = Y2[i2]
            s1 = [s for s in s1 if s != -1]
            s2 = [s for s in s2 if s != -1]
            pi = [p for p in pi if p != -1]

            if len(s1) > size:
                x1 = x1[:size, :, :, :]
                x2 = x2[:size, :, :, :]
                y1 = y1[:size]
                y2 = y2[:size]
                Y1 = Y1[:size]
                Y2 = Y2[:size]
                s1 = s1[:size]
                s2 = s2[:size]
                pi = pi[:size]

            data_dict = {
                "x1": x1,
                "x2": x2,
                "y1": y1.numpy(),
                "y2": y2.numpy(),
                "Y1": Y1,
                "Y2": Y2,
                "s1": s1,
                "s2": s2,
                "p": pi,
                "index_to_class": index_to_network_class,
            }

            save_path = os.path.join(self.args.data_dir, data_file)
            with open(save_path, "wb") as f:
                np.save(f, data_dict)
            print("local data saved at ", save_path)

        data_dict = np.load(save_path, allow_pickle=True).item()
        x1, y1 = data_dict["x1"], data_dict["y1"]
        y1 = torch.from_numpy(y1)
        Y1 = torch.from_numpy(data_dict["Y1"])
        s1 = data_dict["s1"]

        x2, y2 = data_dict["x2"], data_dict["y2"]
        y2 = torch.from_numpy(y2)
        Y2 = torch.from_numpy(data_dict["Y2"])
        s2 = data_dict["s2"]
        pi = data_dict["p"]
        index_to_network_class = data_dict["index_to_class"]

        if validation:
            return x1, y1, Y1, s1, x2, y2, Y2, s2, pi, index_to_network_class

        else:
            gold_label_ratio = np.bincount(np.concatenate((y1.numpy(), y2.numpy())))
            gold_label_ratio = torch.tensor(gold_label_ratio / gold_label_ratio.sum())
            print("gold label ratio:", gold_label_ratio)
            partial_ratio = None

            if self.estimate_margin:
                # reload is_a_graph & base_classes
                is_a_graph = networkx.DiGraph()
                if self.args.num_class == 10:
                    # Load the hierachies for CIFAR10 and populate the graph with edges from subclasses to superclasses
                    is_a_graph = create_is_a_graph(cifar10_theory)
                    base_classes = cifar10_classes
                elif self.args.num_class == 100:
                    # Load the hierachies for CIFAR100 and populate the graph with edges from subclasses to superclasses
                    is_a_graph = create_is_a_graph(cifar100_theory)
                    base_classes = cifar100_classes
                
                count = Counter(s1)
                num_data = sum(count.values())
                partial_ratio = []
                label_vec_to_partial = {}
                partial_class_to_idx = {}
                counted_partial_class = -1

                for i in range(self.args.num_class):
                    for j in range(self.args.num_class):
                        l1, l2 = [
                            [index_to_network_class[i]],
                            [index_to_network_class[j]],
                        ]
                        
                        partial_label, _, _ = get_smallest_common_ancestor(
                            l1, l2, is_a_graph, base_classes
                        )
                        partial_label = partial_label[-1]

                        if partial_label not in partial_class_to_idx:
                            counted_partial_class += 1
                            partial_class_to_idx[partial_label] = counted_partial_class
                            partial_ratio.append(count[partial_label] / num_data)
                        label_vec_to_partial[(i, j)] = partial_class_to_idx[
                            partial_label
                        ]

                print(counted_partial_class + 1)
                label_vec_to_partial["total"] = counted_partial_class + 1
                partial_ratio = torch.tensor(partial_ratio)

                # gold_label_ratio, partial_ratio, label_vec_to_partial = ["label_ratio"], data_dict["partial_ratio"], data_dict["label_vec_to_partial"]

            return (
                x1,
                y1,
                Y1,
                s1,
                x2,
                y2,
                Y2,
                s2,
                pi,
                index_to_network_class,
                gold_label_ratio,
                partial_ratio,
                label_vec_to_partial,
            )

    def loadOriginalTestDataset(self):
        return self.loadOriginalDataset(train=False)
    
    def loadOriginalTrainDataset(self):
        return self.loadOriginalDataset(train=True)
    
    def loadOriginalDataset(self, train=False):
        if self.args.num_class == 10:
            dataset = datasets.CIFAR10(
                root="./data", train=train, transform=self.test_transform, download=True
            )
        elif self.args.num_class == 100:
            dataset = datasets.CIFAR100(
                root="./data", train=train, transform=self.test_transform, download=True
            )
        data, labels, classes, class_to_idx = (
                np.array(dataset.data),
                np.array(dataset.targets),
                dataset.classes,
                dataset.class_to_idx,
            )
        return data, labels, classes, class_to_idx
    

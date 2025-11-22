import torchvision.transforms as transforms
import numpy as np
import torch
import sys
from .loader import DatasetLoader

sys.path.append("../")
from utils_datasets.utils_data import (
    gen_imbalanced_data,
)
import os
import torchvision


class ArithemticLoader(DatasetLoader):
    def __init__(
        self, args, estimate_margin=False, validation_set_size=0, scallop=False
    ):
        DatasetLoader.__init__(
            self, args, estimate_margin, validation_set_size, scallop
        )

    def getDomainRestrictions(self):
        return None
    
    def getIndexToOutputClass(self):
        if self.args.dataset == "mmax":
            domain = 10
        else:
            domain = self.args.M * 9 + 1
        index_to_output_class = {i: i for i in range(domain)}
        return index_to_output_class

    def get_pll_dataset(self, validation=False):
        if self.args.dataset == "mmax":
            func = max
        else:
            func = sum

        if validation:
            size = self.validation_set_size
        else:
            size = self.args.size_partial_dataset
        print("==> Loading local data copy in the long-tailed setup")
        data_file = self.getDataFileName(validation, size)
        save_path = os.path.join(self.args.data_dir, data_file)
        if not os.path.exists(save_path):
            # Create training data
            data, labels, _, _ = self.loadOriginalTrainDataset()
            index_to_network_class = {i: i for i in range(10)}
            train_data, train_labels = gen_imbalanced_data(
                data,
                labels,
                self.args.num_class,
                self.args.imb_type,
                self.args.imb_factor,
            )

            # Sample args.size_partial_dataset images (x)
            rnd = [
                np.random.choice(len(train_data), size=size) for _ in range(self.args.M)
            ]
            # Get the x images
            xs = [train_data[r] for r in rnd]
            # Get the gold labels for x images
            ys = [train_labels[r] for r in rnd]

            s = list()
            # Get the partial label (M-MAX)
            for i in range(size):
                s.append(func([ys[j][i].item() for j in range(self.args.M)]))

            data_dict = {
                "x": xs,
                "y": ys,
                "s": s,
                "index_to_class": index_to_network_class,
            }

            save_path = os.path.join(self.args.data_dir, data_file)
            with open(save_path, "wb") as f:
                np.save(f, data_dict)
            print("local data saved at ", save_path)

        data_dict = np.load(save_path, allow_pickle=True).item()
        xs, ys, ss = data_dict["x"], data_dict["y"], data_dict["s"]
        ys = [y.numpy() for y in ys]
        index_to_network_class = data_dict["index_to_class"]

        if validation:
            return xs, ys, ss
        else:
            gold_label_ratio = np.bincount(np.concatenate(ys[:]))
            gold_label_ratio = torch.tensor(gold_label_ratio / gold_label_ratio.sum())
            print("gold label ratio:", gold_label_ratio)
            partial_ratio = None
            label_vec_to_partial = None
            if self.estimate_margin:
                if self.args.dataset == "mmax":
                    minlength = 10
                elif self.args.dataset == "msum":
                    minlength = self.args.M * 9 + 1
                else:
                    minlength = -1

                partial_ratio = np.bincount(ss, minlength=minlength)
                partial_ratio = torch.tensor(
                    partial_ratio / partial_ratio.sum()
                ).float()

            return (
                xs,
                ys,
                ss,
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
        dataset = torchvision.datasets.MNIST(
            root="./data" + "MNIST",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )
        data, labels, classes, class_to_idx = (
            np.array(dataset.data),
            np.array(dataset.targets),
            dataset.classes,
            dataset.class_to_idx
        )
        return data, labels, classes, class_to_idx



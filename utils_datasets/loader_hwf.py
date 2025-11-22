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
import math

train_root = "/home/experiments/data/HWF/Handwritten_Math_Symbols/train"
test_root = "/home/experiments/data/HWF/Handwritten_Math_Symbols/test"

class HWFLoader(DatasetLoader):
    def __init__(
        self, args, estimate_margin=False, validation_set_size=0, scallop=False
    ):
        DatasetLoader.__init__(
            self, args, estimate_margin, validation_set_size, scallop
        )
        assert args.M % 2 == 1

    def getDomainRestrictions(self):
        # HWF scenario: 
        # if index_to_network_class[k] in [10,11,12] , j % 2 == 0 => max = 0
        # if index_to_network_class[k] in [0--9]     , j % 2 == 1 => max = 0 
        def hwf(c,j):
            if c in [10,11,12] and j % 2 == 0:
                return 0
            if c in [0,1,2,3,4,5,6,7,8,9] and j % 2 == 1:
                return 0
            return 1
        return hwf
    
    def _addParenthesis(self, expression):
        start = 0
        new_expression = ""
        for i in range(len(expression)):
            if i + 1 > 2 and (i + 1) % 2 == 1:
                new_expression = "(" + new_expression + expression[start : i + 1] + ")"
                start = i + 1
        return new_expression

    def getIndexToOutputClass(self):
        # The min value is (0-9)-9*9
        # The max value is (9*9)*9*9
        min_value = -9 ** (math.ceil(self.args.M / 2) - 1) #(-math.floor(self.args.M / 2) * 9) * 9
        max_value = 9 ** math.ceil(self.args.M / 2)
        index_to_output_class = {
            c: i for c, i in enumerate(range(min_value, max_value + 1))
        }
        return index_to_output_class

    def get_train_label_cnt(self, ys):
        # In the HWF formula dataset the classes +,-,* are the first three ones and then the digits follow.
        train_digits_cnt = None
        train_operators_cnt = None
        for index, y in enumerate(ys):
            if index % 2 == 0:
                if train_digits_cnt == None:
                    train_digits_cnt = torch.unique(y, sorted=True, return_counts=True)[
                        -1
                    ]
                else:
                    train_digits_cnt += torch.unique(
                        y, sorted=True, return_counts=True
                    )[-1]
            else:
                if train_operators_cnt == None:
                    train_operators_cnt = torch.unique(
                        y, sorted=True, return_counts=True
                    )[-1]
                else:
                    train_operators_cnt += torch.unique(
                        y, sorted=True, return_counts=True
                    )[-1]
        # train_digits_cnt holds the occurences of digits
        # train_operators_cnt holds the occurences of operators
        # Custom population of the vector train_label_cnt
        train_label_cnt = torch.zeros(13)
        train_label_cnt[0] = train_digits_cnt[0]
        train_label_cnt[1] = train_digits_cnt[1]
        train_label_cnt[2] = train_operators_cnt[0]
        train_label_cnt[3] = train_operators_cnt[1]
        train_label_cnt[4] = train_operators_cnt[2]
        train_label_cnt[5] = train_digits_cnt[2]
        train_label_cnt[6] = train_digits_cnt[3]
        train_label_cnt[7] = train_digits_cnt[4]
        train_label_cnt[8] = train_digits_cnt[5]
        train_label_cnt[9] = train_digits_cnt[6]
        train_label_cnt[10] = train_digits_cnt[7]
        train_label_cnt[11] = train_digits_cnt[8]
        train_label_cnt[12] = train_digits_cnt[9]
        return train_label_cnt

    def get_pll_dataset(self, validation=False):
        if validation:
            size = self.validation_set_size
        else:
            size = self.args.size_partial_dataset
        print("==> Loading local data copy in the long-tailed setup")
        data_file = self.getDataFileName(validation, size)
        save_path = os.path.join(self.args.data_dir, data_file)
        if not os.path.exists(save_path):
            data, labels, _, class_to_idx = self.loadOriginalTrainDataset()
            index_to_network_class = {idx: cls for cls, idx in class_to_idx.items()}

            train_data, train_labels = gen_imbalanced_data(
                data,
                labels,
                self.args.num_class,
                self.args.imb_type,
                self.args.imb_factor,
            )

            ################################################################################
            digits_data = list()
            digits_labels = list()
            symbols_data = list()
            symbols_labels = list()
            for i in range(train_labels.shape[0]):
                if index_to_network_class[train_labels[i].item()] in ["10", "11", "12"]:
                    symbols_data.append(train_data[i])
                    symbols_labels.append(train_labels[i].item())
                else:
                    digits_data.append(train_data[i])
                    digits_labels.append(train_labels[i].item())

            digits_labels = torch.IntTensor(digits_labels)
            symbols_labels = torch.IntTensor(symbols_labels)
            d1 = np.zeros(
                [
                    len(digits_data),
                    digits_data[0].shape[0],
                    digits_data[0].shape[1],
                ]
            )
            for i in range(len(digits_data)):
                d1[i] = digits_data[i]
            digits_data = d1
            d2 = np.zeros(
                [
                    len(symbols_data),
                    symbols_data[0].shape[0],
                    symbols_data[0].shape[1],
                ]
            )
            for i in range(len(symbols_data)):
                d2[i] = symbols_data[i]
            symbols_data = d2
            ################################################################################

            # Sample args.size_partial_dataset images (x)
            rnd = [
                np.random.choice(len(digits_data), size=size)
                if i % 2 == 0
                else np.random.choice(len(symbols_data), size=size)
                for i in range(self.args.M)
            ]

            # Get the x images
            xs = [
                digits_data[rnd[i]] if i % 2 == 0 else symbols_data[rnd[i]]
                for i in range(self.args.M)
            ]

            # Get the gold labels for x images
            ys = [
                digits_labels[rnd[i]] if i % 2 == 0 else symbols_labels[rnd[i]]
                for i in range(self.args.M)
            ]

            s = list()
            # Get the partial label

            ################################################################################
            mask = {
                "0": "0",
                "1": "1",
                "2": "2",
                "3": "3",
                "4": "4",
                "5": "5",
                "6": "6",
                "7": "7",
                "8": "8",
                "9": "9",
                "10": "+",
                "11": "-",
                "12": "*",
            }

            for i in range(size):
                expression = "".join(
                    [
                        mask[index_to_network_class[ys[j][i].item()]]
                        for j in range(self.args.M)
                    ]
                )
                expression = self._addParenthesis(expression)
                s.append(eval(expression))
                if eval(expression) == -288:
                    print()
            ################################################################################

            index_to_network_class = {
                i: int(c) for i, c in index_to_network_class.items()
            }
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
            """
            gold_label_ratio is only for evaluation purpose, and will not be used in training.
            """
            gold_label_ratio = np.bincount(np.concatenate(ys[:]))
            gold_label_ratio = torch.tensor(gold_label_ratio / gold_label_ratio.sum())
            print("gold label ratio:", gold_label_ratio)
            return xs, ys, ss, index_to_network_class, gold_label_ratio, None, None

    def loadOriginalTestDataset(self):
        return self.loadOriginalDataset(train=False)

    def loadOriginalTrainDataset(self):
        return self.loadOriginalDataset(train=True)

    def loadOriginalDataset(self, train=False):
        if train:
            root = train_root
        else:
            root = test_root
        dataset = torchvision.datasets.ImageFolder(
            root=root,
            transform=transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                ]
            ),
        )

        class_to_idx = dataset.class_to_idx
        data = torch.zeros(
            [
                len(dataset),
                dataset[0][0].shape[0],
                dataset[0][0].shape[1],
                dataset[0][0].shape[2],
            ]
        )
        for i in range(len(dataset)):
            data[i] = dataset[i][0]
        data = np.float32(np.array(torch.squeeze(data)))
        labels = np.array(torch.IntTensor(dataset.targets))
        return data, labels, dataset.classes, class_to_idx

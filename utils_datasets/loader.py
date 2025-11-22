import numpy as np
import torch
import sys

sys.path.append("../")
from utils_datasets.mipll_datasets import MIPLL_Dataset, Scallop_Dataset, Gold_Dataset
from utils_datasets.utils_data import (
    gen_imbalanced_data,
)
from utils_datasets.utils_transforms import (
    get_dataset_transforms,
)


class DatasetLoader(object):
    def __init__(
        self, args, estimate_margin=False, validation_set_size=0, scallop=False
    ):
        self.args = args
        self.estimate_margin = estimate_margin
        self.validation_set_size = validation_set_size
        self.scallop = scallop
        self.weak_transform, self.strong_transform, self.test_transform = get_dataset_transforms(self.args)

    def getDomainRestrictions(self):
        return None

    def getIndexToOutputClass(self):
        return None

    def get_pll_dataset(self, validation=False):
        return None

    def loadOriginalTestDataset(self):
        return None
    
    def loadOriginalTrainDataset(self):
        return None

    def get_train_label_cnt(self, ys):
        train_label_cnt = None
        for y in ys:
            if train_label_cnt == None:
                train_label_cnt = torch.unique(y, sorted=True, return_counts=True)[-1]
            else:
                train_label_cnt += torch.unique(y, sorted=True, return_counts=True)[-1]
        return train_label_cnt

    def createTestDataset(self) :
        test_data, test_labels, _, _ = self.loadOriginalTestDataset()
        if self.args.imb_test:        
            print(
                "Using imbalanced test set (training distribution = test distribution)."
            )
            test_data, test_labels = gen_imbalanced_data(
                test_data,
                test_labels,
                self.args.num_class,
                self.args.imb_type,
                self.args.imb_factor,
            )
            test_label_ratio = np.bincount(test_labels)
            test_label_ratio = torch.tensor(test_label_ratio / test_label_ratio.sum())
            print("test label ratio: ", test_label_ratio)
        test_dataset = Gold_Dataset(test_data, test_labels, self.test_transform)
        return test_dataset
    
    def getDataFileName(self, validation, size):
        data_file = (
            "{ds}_val_{val}_imb_{it}_{imf}_sd{sd}_ni={ni}_M={m}.npy".format(
                ds=self.args.dataset,
                val=validation,
                it=self.args.imb_type,
                imf=self.args.imb_ratio,
                sd=self.args.seed,
                ni=size,
                m=self.args.M,
            )
        )
        return data_file

    def load(self):
        index_to_output_class = self.getIndexToOutputClass()
        output_class_to_index = {c: i for i, c in index_to_output_class.items()}
        (
            xs,
            ys,
            ss,
            index_to_network_class,
            gold_label_ratio,
            partial_ratio,
            label_vec_to_partial,
        ) = self.get_pll_dataset()

        ys = [torch.from_numpy(y) for y in ys]
        init_label_dist = torch.ones(self.args.num_class) / self.args.num_class
        # Be careful. If you choose not many partial training samples, then it is likely that not all classes are sampled,
        # especially, when specifying argument 'exp'.
        # Hence, this line of code might return an exception.

        train_label_cnt = self.get_train_label_cnt(ys)
        # train_label_cnt is also used for intialize Acc-shot object

        if self.scallop:
            train_pll_dataset = Scallop_Dataset(
                xs,
                [y.float() for y in ys],
                [output_class_to_index[s] for s in ss],
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
            train_pll_dataset = MIPLL_Dataset(
                xs,
                [y.float() for y in ys],
                ss,
                self.args.size_partial_dataset,
                self.weak_transform,
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

        # estimation loader for distribution estimation
        est_dataset = MIPLL_Dataset(
            xs,
            [y.float() for y in ys],
            ss,
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
            xs_val, ys_val, ss_val = self.get_pll_dataset(validation=True)
            ys_val = [torch.from_numpy(y) for y in ys_val]
            val_dataset = MIPLL_Dataset(
                xs_val,
                [y.float() for y in ys_val],
                ss_val,
                self.validation_set_size,
                self.weak_transform,
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
            label_vec_to_partial
        )

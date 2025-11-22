from torch.utils.data import Dataset
import torch

class MIPLL_Dataset(Dataset):
    def __init__(
        self, x, y, s, l, weak_transform
    ):
        self.x = x
        self.y = y
        self.s = s
        self.l = l
        self.M = len(y)
        self.weak_transform = weak_transform

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        
        each_x_w = [self.weak_transform(self.x[i][index]) for i in range(self.M)]
        each_y = [self.y[i][index] for i in range(self.M)]
        each_s = self.s[index]
        
        return (
            each_x_w,
            each_y,
            each_s,
            index,
        )
    
class Scallop_Dataset(Dataset):
    def __init__(
        self, x, y, s, l, weak_transform
    ):
        self.x = x
        self.y = y
        self.s = s
        self.l = l
        self.M = len(y)
        self.weak_transform = weak_transform

    def __len__(self):
        return self.l

    def __getitem__(self, index):
        
        each_x_w = [self.weak_transform(self.x[i][index]) for i in range(self.M)]
        each_s = self.s[index]
        return each_x_w + [each_s]
    
    @staticmethod
    def collate_fn(batch):
        L = len(batch[0])
        columns = [torch.stack([item[i] for item in batch]) for i in range(L - 1)]
        ys = torch.stack([torch.tensor(item[L - 1]).long() for item in batch])
        return (columns, ys)
    
class Gold_Dataset(Dataset):
    def __init__(
        self, x, y, transform
    ):
        self.data = x
        self.targets = y
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.transform(self.data[index]), int(self.targets[index])
        return img, target
    
class SMALLEST_PARENT_Dataset(Dataset):
    def __init__(
        self, x1, x2, Y1, Y2, y1, y2, s1, s2, p, weak_transform, strong_transform=None
    ):
        self.x1 = x1
        self.Y1 = Y1
        self.y1 = y1
        self.x2 = x2
        self.Y2 = Y2
        self.y2 = y2
        self.s1 = s1
        self.s2 = s2
        self.p = p
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.y1)

    def __getitem__(self, index):
        each_x1_w = self.weak_transform(self.x1[index])
        if self.strong_transform is not None:
            each_x1_s = self.strong_transform(self.x1[index])
        each_Y1 = self.Y1[index]
        each_y1 = self.y1[index]
        each_s1 = self.s1[index]

        each_x2_w = self.weak_transform(self.x2[index])
        if self.strong_transform is not None:
            each_x2_s = self.strong_transform(self.x2[index])
        each_Y2 = self.Y2[index]
        each_y2 = self.y2[index]
        each_s2 = self.s2[index]

        each_p = self.p[index]
        if self.strong_transform is not None:
            return (
                each_x1_w,
                each_x1_s,
                each_x2_w,
                each_x2_s,
                each_Y1,
                each_Y2,
                each_y1,
                each_y2,
                each_s1,
                each_s2,
                each_p,
                index,
            )
        else:
            return (
                each_x1_w,
                each_x2_w,
                each_Y1,
                each_Y2,
                each_y1,
                each_y2,
                each_s1,
                each_s2,
                each_p,
            )

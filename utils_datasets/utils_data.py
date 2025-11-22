import torch
from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np


def gen_imbalanced_data(
    data, targets, num_class, imb_type="exp", imb_factor=0.01, is_cub=False
):
    if imb_type == "original":
        return data, torch.Tensor(targets).long()

    img_max = len(data) / num_class
    img_num_per_cls = get_img_num_per_cls(num_class, imb_type, imb_factor, img_max)

    new_data = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)
    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        the_img_num = len(selec_idx)
        if is_cub:
            new_data += [data[t] for t in selec_idx]
        else:
            new_data.append(data[selec_idx, ...])
        new_targets.extend(
            [
                the_class,
            ]
            * the_img_num
        )
    if not is_cub:
        new_data = np.vstack(new_data)

    new_targets = torch.Tensor(new_targets).long()
    return new_data, new_targets


def get_img_num_per_cls(cls_num, imb_type, imb_factor, img_max):
    img_num_per_cls = []
    if imb_type == "exp":
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == "expr":
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** ((cls_num - cls_idx) / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == "step":
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        raise NotImplementedError("You have chosen an unsupported imb type.")
    return img_num_per_cls


def get_transition_matrix(K, partial_rate):
    transition_matrix = np.zeros((K, K))
    for i in range(K):
        transition_matrix[i, i] = 1
        for j in range(K):
            if j != i:
                transition_matrix[i, j] = -partial_rate * i

    return transition_matrix


def generate_label_dependent_cv_candidate_labels(train_labels, partial_rate):
    if torch.min(train_labels) > 1:
        raise RuntimeError("testError")
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = get_transition_matrix(K, partial_rate)
    print("==> Transition Matrix:")
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K):  # for each class
            if jj == train_labels[j]:  # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError("testError")
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix = np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = p_1
    print("==> Transition Matrix:")
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy(
            (random_n[j, :] < transition_matrix[train_labels[j], :]) * 1
        )

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def unpickle(file):
    with open(file, "rb") as fo:
        res = pickle.load(fo, encoding="bytes")
    return res


def binarize_class(y):
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories="auto")
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)
    label = torch.from_numpy(label)
    return label

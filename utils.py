import os
import torch
import numpy as np


def get_datadir(root, dataset):
    if dataset == 'sun':
        return os.path.join(root, 'SUN')
    elif dataset == 'cub':
        return os.path.join(root, 'CUB')
    elif dataset == 'awa1':
        return os.path.join(root, 'AWA1')
    elif dataset == 'flo':
        return os.path.join(root, 'FLO')


def get_data_stats(dataset, validation=False):
    ''' Basic stats
    Number of categories in the training and testing sets,
    and the number of attributes
    '''
    if dataset == 'sun':
        if validation:
            return 580, 65, 72, 102
        else:
            return 645, 72, 102
    elif dataset == 'cub':
        if validation:
            return 100, 50, 50, 312
        else:
            return 150, 50, 312
    elif dataset == 'awa1':
        if validation:
            return 27, 13, 10, 85
        else:
            return 40, 10, 85
    elif dataset == 'flo':
        if validation:
            return 62, 20, 20, 1024
        else:
            return 82, 20, 1024


def normalize_data(data, mean=None, std=None, max_value=None, min_value=None):
    tmp_data = data.copy()
    if mean is None:
        mean = np.mean(data, axis=0)
        tmp_data -= mean
    if std is None:
        std = np.sqrt(np.mean(tmp_data ** 2, axis=0))
        tmp_data /= std
    if max_value is None:
        max_value = np.max(tmp_data)
    if min_value is None:
        min_value = np.min(tmp_data)

    norm_data = ((data - mean) / std)
    norm_data = (norm_data - min_value) / (max_value - min_value)
    return norm_data, mean, std, max_value, min_value


def labels_mapping(labels):
    mapping = {}
    unique_labels = np.unique(labels)
    mapping['direct'] = dict(zip(unique_labels, np.arange(len(unique_labels))))
    mapping['indirect'] = dict(zip(np.arange(len(unique_labels)), unique_labels))
    new_labels = np.asarray([mapping['direct'][i] for i in labels])
    return new_labels, mapping


def frobenius_norm(x):
     return x.abs().pow(2).sum()


def layer_norm(x, eps=1e-6):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, unbiased=False, keepdim=True)
    return (x - mean) / (std + eps)


def L2norm(x):
    return x / x.norm(p=2, dim=1)[:, None]


def cosine_similarity(x, y=None, eps=1e-8):
    if y is None:
        w = x.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x, x.t()) / (w * w.t()).clamp(min=eps)
    else:
        xx = L2norm(x)
        yy = L2norm(y)
        return xx.matmul(yy.t())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

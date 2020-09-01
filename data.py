import argparse
import os
import ipdb
import numpy as np
import scipy.io as sio
import pandas as pd
import torch
import pickle
from utils import get_datadir, labels_mapping, get_data_stats
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from torch.utils.data import Dataset


BASE = ['X', 'Y', 'emb', 'mapping', 'classnames']

TRAIN = ['train_X', 'train_Y',
         'train_emb', 'train_mapping', 'train_classnames']
VAL_SEEN = ['val_seen_X', 'val_seen_Y',
             'val_seen_emb', 'val_seen_mapping', 'val_seen_classnames']
VAL_UNSEEN = ['val_unseen_X', 'val_unseen_Y',
               'val_unseen_emb', 'val_unseen_mapping',
               'val_unseen_classnames']
TRAINVAL = ['trainval_X', 'trainval_Y',
            'trainval_emb', 'trainval_mapping', 'trainval_classnames']
TEST_SEEN = ['test_seen_X', 'test_seen_Y',
             'test_seen_emb', 'test_seen_mapping', 'test_seen_classnames']
TEST_UNSEEN = ['test_unseen_X', 'test_unseen_Y',
               'test_unseen_emb', 'test_unseen_mapping',
               'test_unseen_classnames']


def sample_imgs(y, idx, ratio=0.2):
    y_unique, y_counts = np.unique(y, return_counts=True)
    idx_val = []
    for val, n_per_class in zip(y_unique, y_counts):
        cond = np.squeeze(np.argwhere(y == val))
        subset = np.random.choice(idx[cond],
                                  replace=False,
                                  size=int(n_per_class*ratio))
        idx_val.extend(subset)
    idx_train = idx[np.isin(idx, idx_val, invert=True)]
    return idx_train, np.asarray(idx_val)


def create_train_val_splits(y_trainval, idx_trainval, n_unseen,
                            seed=123):
    np.random.seed(seed)

    # pick unseen categories
    y_unique = np.unique(y_trainval)
    y_val_unseen = np.random.choice(y_unique, replace=False, size=n_unseen)

    # get val unseen
    cond = np.isin(y_trainval, y_val_unseen)
    idx_val_unseen = idx_trainval[cond]

    y_seen = y_unique[~np.isin(y_unique, y_val_unseen)]
    cond = np.isin(y_trainval, y_seen)
    idx_seen = idx_trainval[cond]

    idx_train, idx_val_seen = sample_imgs(y_trainval[cond], idx_seen)

    return idx_train, idx_val_seen, idx_val_unseen


def load_data_gan(path_gan, path_embedding,
                  val_classes=10,
                  path_sentences=None,
                  normalize=False):
    '''Load features and embedding

    Returns dict
    split_X: (n, 2048) array
    split_Y: (n) array
        from 0 to C_split classes
    split_emb: (A, C_split) array
        attribute representation for each class in the split
    split_mapping: dict
        `direct` class index mapping from full to split
        `indirect` class index mapping from split to full
    split_classnames: dict
        class name according to the full class index

    Note: the test seen split is leaking into train and val splits
          MUST use trainval split only
          or create our own train and val splits
          rather using than those from Xian et al, TPAMI 2018
    '''
    feat = np.load(path_gan)
    if 'train_Y' in feat.files:
        y = feat['train_Y']
    else:
        y = np.concatenate((feat['train_seen_Y'], feat['train_unseen_Y']))

    get_sentences = False
    if path_sentences is not None:
        get_sentences = True
        with open(path_sentences, 'rb') as f:
            sentences = pickle.load(f)['sentences'].T

    embedding = sio.loadmat(path_embedding)
    if 'allclasses_names' in embedding.keys():
        names = embedding['allclasses_names']
    else:
        names =[[[str(i)]] for i in range(embedding['att'].shape[1])]
    full_classnames = classnames_parser(y, names)

    # train and validation merged
    if 'train_X' in feat.files:
        trainval_X, trainval_Y_orig = feat['train_X'], feat['train_Y']
    else:
        trainval_X = np.concatenate((feat['train_seen_X'],
                                     feat['train_unseen_X']))
        trainval_Y_orig = np.concatenate((feat['train_seen_Y'],
                                          feat['train_unseen_Y']))

    trainval_Y, trainval_mapping = labels_mapping(trainval_Y_orig)
    trainval_emb = embedding['att'][:, np.unique(trainval_Y_orig)]
    trainval_classnames = classnames_parser(trainval_Y_orig, names)

    # testing splits
    test_seen_X, test_seen_Y_orig = feat['test_seen_X'], feat['test_seen_Y']
    test_seen_Y, test_seen_mapping = labels_mapping(test_seen_Y_orig)
    test_seen_emb = embedding['att'][:, np.unique(test_seen_Y_orig)]
    test_seen_classnames = classnames_parser(test_seen_Y_orig, names)

    test_unseen_X, test_unseen_Y_orig = feat['test_unseen_X'], feat['test_unseen_Y']
    test_unseen_Y, test_unseen_mapping = labels_mapping(test_unseen_Y_orig)
    test_unseen_emb = embedding['att'][:, np.unique(test_unseen_Y_orig)]
    test_unseen_classnames = classnames_parser(test_unseen_Y_orig, names)

    # class-attribute matrix for all classes (used in GZSL)
    full_emb = embedding['att']

    if normalize:
        scaler = preprocessing.MinMaxScaler()
        trainval_X = scaler.fit_transform(trainval_X)
        test_seen_X = scaler.transform(test_seen_X)
        test_unseen_X = scaler.transform(test_unseen_X)

    if get_sentences:
        trainval_sentences = sentences[:, np.unique(trainval_Y_orig)]
        test_seen_sentences = sentences[:, np.unique(test_seen_Y_orig)]
        test_unseen_sentences = sentences[:, np.unique(test_unseen_Y_orig)]

        return {# trainval split
                'trainval_X': trainval_X.astype(np.float32),
                'trainval_Y': trainval_Y,
                'trainval_emb': trainval_emb.astype(np.float32),
                'trainval_mapping': trainval_mapping,
                'trainval_classnames': trainval_classnames,
                'trainval_sentences': trainval_sentences.astype(np.float32),
                # 2 test splits
                'test_seen_X': test_seen_X.astype(np.float32),
                'test_seen_Y': test_seen_Y,
                'test_seen_emb': test_seen_emb.astype(np.float32),
                'test_seen_mapping': test_seen_mapping,
                'test_seen_classnames': test_seen_classnames,
                'test_seen_sentences': test_seen_sentences.astype(np.float32),
                'test_unseen_X': test_unseen_X.astype(np.float32),
                'test_unseen_Y': test_unseen_Y,
                'test_unseen_emb': test_unseen_emb.astype(np.float32),
                'test_unseen_mapping': test_unseen_mapping,
                'test_unseen_classnames': test_unseen_classnames,
                'test_unseen_sentences': test_unseen_sentences.astype(np.float32),
                # full cam
                'test_seen_Y_orig': test_seen_Y_orig.astype(np.int),
                'test_unseen_Y_orig': test_unseen_Y_orig.astype(np.int),
                'full_emb': full_emb.astype(np.float32),
                'full_classnames': full_classnames,
                'full_sentences': sentences.astype(np.float32),
                'seen_Y': np.unique(test_seen_Y_orig.astype(np.int)),
                'unseen_Y': np.unique(test_unseen_Y_orig.astype(np.int))}

    return {# trainval split
            'trainval_X': trainval_X.astype(np.float32),
            'trainval_Y': trainval_Y,
            'trainval_emb': trainval_emb.astype(np.float32),
            'trainval_mapping': trainval_mapping,
            'trainval_classnames': trainval_classnames,
            # 2 test splits
            'test_seen_X': test_seen_X.astype(np.float32),
            'test_seen_Y': test_seen_Y,
            'test_seen_emb': test_seen_emb.astype(np.float32),
            'test_seen_mapping': test_seen_mapping,
            'test_seen_classnames': test_seen_classnames,
            'test_unseen_X': test_unseen_X.astype(np.float32),
            'test_unseen_Y': test_unseen_Y,
            'test_unseen_emb': test_unseen_emb.astype(np.float32),
            'test_unseen_mapping': test_unseen_mapping,
            'test_unseen_classnames': test_unseen_classnames,
            # full cam
            'test_seen_Y_orig': test_seen_Y_orig.astype(np.int),
            'test_unseen_Y_orig': test_unseen_Y_orig.astype(np.int),
            'full_emb': full_emb.astype(np.float32),
            'full_classnames': full_classnames,
            'seen_Y': np.unique(test_seen_Y_orig.astype(np.int)),
            'unseen_Y': np.unique(test_unseen_Y_orig.astype(np.int))}


def load_data_separate(path_features, path_embedding,
                       val_classes=10,
                       path_sentences=None,
                       normalize=False):
    features = sio.loadmat(path_features)
    x = features['features'].T
    y = features['labels'] - 1  # labels start at 1, need to subtract 1

    get_sentences = False
    if path_sentences is not None:
        get_sentences = True
        with open(path_sentences, 'rb') as f:
            sentences = pickle.load(f)['sentences'].T

    embedding = sio.loadmat(path_embedding)
    if 'allclasses_names' in embedding.keys():
        names = embedding['allclasses_names']
    else:
        names =[[[str(i)]] for i in range(embedding['att'].shape[1])]
    full_classnames = classnames_parser(y, names)

    # train and validation merged
    idx = np.squeeze(embedding['trainval_loc']) - 1
    trainval_X, trainval_Y_orig = x[idx, :], np.squeeze(y[idx])
    trainval_Y, trainval_mapping = labels_mapping(trainval_Y_orig)
    trainval_emb = embedding['att'][:, np.unique(trainval_Y_orig)]
    trainval_classnames = classnames_parser(trainval_Y_orig, names)

    # new training and validation splits
    # with 0.8 train_seen and val_seen
    idx_train, idx_val_seen, idx_val_unseen = create_train_val_splits(
        trainval_Y_orig, idx, val_classes)

    # training split
    train_X, train_Y_orig = x[idx_train, :], np.squeeze(y[idx_train])
    train_Y, train_mapping = labels_mapping(train_Y_orig)
    train_emb = embedding['att'][:, np.unique(train_Y_orig)]
    train_classnames = classnames_parser(train_Y_orig, names)

    # validation splits
    val_seen_X, val_seen_Y_orig = x[idx_val_seen, :], np.squeeze(y[idx_val_seen])
    val_seen_Y, val_seen_mapping = labels_mapping(val_seen_Y_orig)
    val_seen_emb = embedding['att'][:, np.unique(val_seen_Y_orig)]
    val_seen_classnames = classnames_parser(val_seen_Y_orig, names)

    val_unseen_X, val_unseen_Y_orig = x[idx_val_unseen, :], np.squeeze(y[idx_val_unseen])
    val_unseen_Y, val_unseen_mapping = labels_mapping(val_unseen_Y_orig)
    val_unseen_emb = embedding['att'][:, np.unique(val_unseen_Y_orig)]
    val_unseen_classnames = classnames_parser(val_unseen_Y_orig, names)

    # testing splits
    idx = np.squeeze(embedding['test_seen_loc']) - 1
    test_seen_X, test_seen_Y_orig = x[idx, :], np.squeeze(y[idx])
    test_seen_Y, test_seen_mapping = labels_mapping(test_seen_Y_orig)
    test_seen_emb = embedding['att'][:, np.unique(test_seen_Y_orig)]
    test_seen_classnames = classnames_parser(test_seen_Y_orig, names)

    idx = np.squeeze(embedding['test_unseen_loc']) - 1
    test_unseen_X, test_unseen_Y_orig = x[idx, :], np.squeeze(y[idx])
    test_unseen_Y, test_unseen_mapping = labels_mapping(test_unseen_Y_orig)
    test_unseen_emb = embedding['att'][:, np.unique(test_unseen_Y_orig)]
    test_unseen_classnames = classnames_parser(test_unseen_Y_orig, names)

    # class-attribute matrix for all classes (used in GZSL)
    full_emb = embedding['att']

    if normalize:
        scaler = preprocessing.MinMaxScaler()
        trainval_X = scaler.fit_transform(trainval_X)
        test_seen_X = scaler.transform(test_seen_X)
        test_unseen_X = scaler.transform(test_unseen_X)

    if get_sentences:
        trainval_sentences = sentences[:, np.unique(trainval_Y_orig)]
        train_sentences = sentences[:, np.unique(train_Y_orig)]
        val_seen_sentences = sentences[:, np.unique(val_seen_Y_orig)]
        val_unseen_sentences = sentences[:, np.unique(val_unseen_Y_orig)]
        test_seen_sentences = sentences[:, np.unique(test_seen_Y_orig)]
        test_unseen_sentences = sentences[:, np.unique(test_unseen_Y_orig)]

        return {# training split
                'train_X': train_X.astype(np.float32),
                'train_Y': train_Y,
                'train_emb': train_emb.astype(np.float32),
                'train_mapping': train_mapping,
                'train_classnames': train_classnames,
                'train_sentences': train_sentences.astype(np.float32),
                # trainval split
                'trainval_X': trainval_X.astype(np.float32),
                'trainval_Y': trainval_Y,
                'trainval_emb': trainval_emb.astype(np.float32),
                'trainval_mapping': trainval_mapping,
                'trainval_classnames': trainval_classnames,
                'trainval_sentences': trainval_sentences.astype(np.float32),
                # 2 validation splits
                'val_seen_X': val_seen_X.astype(np.float32),
                'val_seen_Y': val_seen_Y,
                'val_seen_emb': val_seen_emb.astype(np.float32),
                'val_seen_mapping': val_seen_mapping,
                'val_seen_classnames': val_seen_classnames,
                'val_seen_sentences': val_seen_sentences.astype(np.float32),
                'val_unseen_X': val_unseen_X.astype(np.float32),
                'val_unseen_Y': val_unseen_Y,
                'val_unseen_emb': val_unseen_emb.astype(np.float32),
                'val_unseen_mapping': val_unseen_mapping,
                'val_unseen_classnames': val_unseen_classnames,
                'val_unseen_sentences': val_unseen_sentences.astype(np.float32),
                # 2 test splits
                'test_seen_X': test_seen_X.astype(np.float32),
                'test_seen_Y': test_seen_Y,
                'test_seen_emb': test_seen_emb.astype(np.float32),
                'test_seen_mapping': test_seen_mapping,
                'test_seen_classnames': test_seen_classnames,
                'test_seen_sentences': test_seen_sentences.astype(np.float32),
                'test_unseen_X': test_unseen_X.astype(np.float32),
                'test_unseen_Y': test_unseen_Y,
                'test_unseen_emb': test_unseen_emb.astype(np.float32),
                'test_unseen_mapping': test_unseen_mapping,
                'test_unseen_classnames': test_unseen_classnames,
                'test_unseen_sentences': test_unseen_sentences.astype(np.float32),
                # full cam
                'val_seen_Y_orig': val_seen_Y_orig.astype(np.int),
                'val_unseen_Y_orig': val_unseen_Y_orig.astype(np.int),
                'test_seen_Y_orig': test_seen_Y_orig.astype(np.int),
                'test_unseen_Y_orig': test_unseen_Y_orig.astype(np.int),
                'full_emb': full_emb.astype(np.float32),
                'full_classnames': full_classnames,
                'full_sentences': sentences.astype(np.float32)}

    return {# training split
            'train_X': train_X.astype(np.float32),
            'train_Y': train_Y,
            'train_emb': train_emb.astype(np.float32),
            'train_mapping': train_mapping,
            'train_classnames': train_classnames,
            # trainval split
            'trainval_X': trainval_X.astype(np.float32),
            'trainval_Y': trainval_Y,
            'trainval_emb': trainval_emb.astype(np.float32),
            'trainval_mapping': trainval_mapping,
            'trainval_classnames': trainval_classnames,
            # 2 validation splits
            'val_seen_X': val_seen_X.astype(np.float32),
            'val_seen_Y': val_seen_Y,
            'val_seen_emb': val_seen_emb.astype(np.float32),
            'val_seen_mapping': val_seen_mapping,
            'val_seen_classnames': val_seen_classnames,
            'val_unseen_X': val_unseen_X.astype(np.float32),
            'val_unseen_Y': val_unseen_Y,
            'val_unseen_emb': val_unseen_emb.astype(np.float32),
            'val_unseen_mapping': val_unseen_mapping,
            'val_unseen_classnames': val_unseen_classnames,
            # 2 test splits
            'test_seen_X': test_seen_X.astype(np.float32),
            'test_seen_Y': test_seen_Y,
            'test_seen_emb': test_seen_emb.astype(np.float32),
            'test_seen_mapping': test_seen_mapping,
            'test_seen_classnames': test_seen_classnames,
            'test_unseen_X': test_unseen_X.astype(np.float32),
            'test_unseen_Y': test_unseen_Y,
            'test_unseen_emb': test_unseen_emb.astype(np.float32),
            'test_unseen_mapping': test_unseen_mapping,
            'test_unseen_classnames': test_unseen_classnames,
            # full cam
            'val_seen_Y_orig': val_seen_Y_orig.astype(np.int),
            'val_unseen_Y_orig': val_unseen_Y_orig.astype(np.int),
            'test_seen_Y_orig': test_seen_Y_orig.astype(np.int),
            'test_unseen_Y_orig': test_unseen_Y_orig.astype(np.int),
            'full_emb': full_emb.astype(np.float32),
            'full_classnames': full_classnames}


def classnames_parser(labels, names):
    classnames = [n[0][0] for n in names]
    classnames = dict(zip(np.arange(len(classnames)), classnames))

    # note that labels have to start at 0 (not 1)
    unique_labels = np.unique(labels)

    classmapping = {}
    for l in unique_labels:
        classmapping[l] = classnames[l]
    return classmapping


class AttributeDataset(Dataset):
    def __init__(self, root, dataset,
                 features_path=None,
                 mode='trainval',
                 sentences=False,
                 generalized=False,
                 normalize=False,
                 both=False,
                 on_gpu=False):
        data_dir = get_datadir(root, dataset)
        if features_path is None:
            features_path = os.path.join(data_dir, 'res101.mat')
            load_data = load_data_separate
        else:
            load_data = load_data_gan
        meta_path = os.path.join(data_dir, 'att_splits.mat')

        _, n_val_classes, _, _ = get_data_stats(dataset, validation=True)

        if sentences:
            # works for CUB dataset
            sentences_path = os.path.join(data_dir, 'CUB_supporting_data.p')
            data = load_data(
                features_path, meta_path, n_val_classes,
                sentences_path, normalize=normalize)
        else:
            data = load_data(
                features_path, meta_path, n_val_classes,
                normalize=normalize)

        if mode == 'train':
            keys = TRAIN
        elif mode == 'val_seen':
            keys = VAL_SEEN
        elif mode == 'val_unseen':
            keys = VAL_UNSEEN
        elif mode == 'trainval':
            keys = TRAINVAL
        elif mode == 'test_seen':
            keys = TEST_SEEN
        elif mode == 'test_unseen':
            keys = TEST_UNSEEN

        self.data = {}
        if sentences:
            kk = keys + [mode + '_sentences']
            bb = BASE + ['sentences']
            for k, b in zip(kk, bb):
                if b in ['X', 'Y'] and on_gpu:
                    self.data[b] = torch.from_numpy(data[k]).cuda()
                else:
                    self.data[b] = data[k]
        else:
            for k, b in zip(keys, BASE):
                if b in ['X', 'Y'] and on_gpu:
                    self.data[b] = torch.from_numpy(data[k]).cuda()
                else:
                    self.data[b] = data[k]

        if generalized:
            self.data['Y_orig'] = data[mode + '_Y_orig']
            self.data['full_emb'] = data['full_emb']
            if sentences:
                self.data['full_sentences'] = data['full_sentences']

        self.mode = mode
        self.generalized = generalized
        self.both = both
        if self.both:
            self.data['seen_Y'] = data['seen_Y']
            self.data['unseen_Y'] = data['unseen_Y']

    def __getitem__(self, index):
        feat = self.data['X'][index, :]
        label = self.data['Y'][index]

        if self.generalized:
            label_g = self.data['Y_orig'][index]
            return feat, (label, label_g)
        else:
            if self.both:
                if label in self.data['seen_Y']:
                    seen = np.float32(1.)
                    unseen = np.float32(0.)
                elif label in self.data['unseen_Y']:
                    seen = np.float32(0.)
                    unseen = np.float32(1.)
                return feat, (label, seen, unseen)
            else:
                return feat, label

    def __len__(self):
        return self.data['X'].shape[0]

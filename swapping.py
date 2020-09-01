import os
import sys
import argparse
import numpy as np
import matplotlib.pylab as plt
from validate import calibrate
from sklearn import linear_model as lm


parser = argparse.ArgumentParser(description='PyTorch DML')
parser.add_argument('--att', type=str, required=True,
                    help='path to scores with attributes')
parser.add_argument('--sen', type=str, required=True,
                    help='path to scores without sentences')
args = parser.parse_args()


def L2norm(x):
    return x / np.linalg.norm(x, axis=1)[:, None]


def get_probs(feat, proxies, temp=0.05):
    diff = np.dot(L2norm(feat), L2norm(proxies).T)
    num = np.exp(diff / temp)
    den = num.sum(1) + 1e-8
    return num / den[:, None]


def classify(data, proxies):
    probs_seen = get_probs(data['feats_seen'], proxies)
    probs_unseen = get_probs(data['feats_unseen'], proxies)

    calibrate(probs_seen, probs_unseen,
              data['y_seen'], data['y_unseen'],
              data['seen_idx'])


def merge(emb_seen, emb_unseen, seen_idx):
    n = len(emb_seen) + len(emb_unseen)
    d = emb_seen.shape[1]
    emb = np.zeros((n, d))

    seen_idx_count = 0
    unseen_idx_count = 0
    for i in range(n):
        if i in seen_idx:
            emb[i, :] = emb_seen[seen_idx_count, :]
            seen_idx_count += 1
        else:
            emb[i, :] = emb_unseen[unseen_idx_count, :]
            unseen_idx_count += 1
    return emb


def main():
    data_att = np.load(args.att)
    data_sen = np.load(args.sen)

    print('\nTrain: attributes, Test: attributes')
    classify(data_att, data_att['emb_full'])

    print('\nTrain: sentences, Test: sentences')
    classify(data_sen, data_sen['emb_full'])

    model = lm.Ridge(alpha=0.1, normalize=True).fit(data_att['emb_seen'], data_sen['emb_seen'])
    pred_unseen_emb = model.predict(data_att['emb_unseen'])
    emb = merge(data_sen['emb_seen'], pred_unseen_emb, data_sen['seen_idx'])
    print('\nTrain: sentences, Test: attributes')
    classify(data_sen, emb)

    model = lm.Ridge(alpha=0.1, normalize=True).fit(data_sen['emb_seen'], data_att['emb_seen'])
    pred_unseen_emb = model.predict(data_sen['emb_unseen'])
    emb = merge(data_att['emb_seen'], pred_unseen_emb, data_att['seen_idx'])
    print('\nTrain: attributes, Test: sentences')
    classify(data_att, emb)


if __name__ == '__main__':
    main()

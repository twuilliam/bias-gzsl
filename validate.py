import torch
import numpy as np


def calibrated_stacking(scores, idx, gamma=0):
    sc = scores.copy()
    sc[:, idx] -= gamma
    return sc


def average_harmonic_mean(seen, unseen, v=True):
    '''Compute the harmonic mean
    seen and unseen are dictionaries
    '''
    acc_seen = average_accuracy_score(seen['y_true'], seen['y_pred'])
    acc_unseen = average_accuracy_score(unseen['y_true'], unseen['y_pred'])
    harmonic_mean = (2 * (acc_seen * acc_unseen) / (acc_seen + acc_unseen))
    if v:
        print('Avg per-class accuracy score (seen classes): %.2f' % (acc_seen * 100))
        print('Avg per-class accuracy score (unseen classes): %.2f' % (acc_unseen * 100))
        print('(Avg per-class) Harmonic mean: %.2f' %
              (harmonic_mean * 100))
    else:
        return acc_seen, acc_unseen, harmonic_mean


def average_accuracy_score(y_true, y_pred):
    '''Per-class accuracy'''
    unique_labels = np.unique(y_true)
    acc = []
    for label in unique_labels:
        n = np.count_nonzero(y_true == label)
        correct_preds = np.count_nonzero(
            np.logical_and(y_true == label, y_pred == label))
        acc.append(correct_preds / float(n))
    return np.mean(acc)


def calibrate(sf, uf, y_true_st, y_true_ut, idx, v=None):

    acc_seen = []
    acc_unseen = []
    acc_hm = []
    gammas = []

    for gamma in np.arange(-1, 1., 0.01):
        seen_scores = calibrated_stacking(sf, idx, gamma)
        unseen_scores = calibrated_stacking(uf, idx, gamma)

        y_pred_gzsl_test_seen = np.argmax(seen_scores, axis=1)
        y_pred_gzsl_test_unseen = np.argmax(unseen_scores, axis=1)

        a_s, a_u, hm = average_harmonic_mean(
            {'y_true': y_true_st,
             'y_pred': y_pred_gzsl_test_seen},
            {'y_true': y_true_ut,
             'y_pred': y_pred_gzsl_test_unseen}, v=False)

        acc_seen.append(a_s)
        acc_unseen.append(a_u)
        acc_hm.append(hm)
        gammas.append(gamma)

    idx = np.argmax(acc_hm)
    txt = ('Validation -- (HM) %.2f\t(SEEN) %.2f\t(UNSEEN) %.2f\t(Gamma): %.2f' %
           (acc_hm[idx] * 100,
            acc_seen[idx] * 100,
            acc_unseen[idx] * 100,
            gammas[idx]))

    if v is not None:
        v(txt)
    print(txt)

    return gammas[idx]

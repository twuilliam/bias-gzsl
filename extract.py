import os
import json
import argparse
import torch
import numpy as np
from data import AttributeDataset
from models import LinearProjection, MLP, Base
from models import ProxyNet, ProxyLoss
from validate import calibrate


# Training settings
parser = argparse.ArgumentParser(description='PyTorch DML')
parser.add_argument('--model-path', type=str, default='exp', metavar='ED',
                    help='model path')

args = parser.parse_args()
path = os.path.dirname(args.model_path)
with open(os.path.join(path, 'config.json')) as f:
    tmp = json.load(f)

tmp['model_path'] = args.model_path
args = type('parser', (object,), tmp)


def main():
    test_seen_loader = torch.utils.data.DataLoader(
        AttributeDataset(args.data_dir, args.dataset,
                         features_path=args.gan_path,
                         mode='test_seen',
                         generalized=True,
                         normalize=args.normalize,
                         sentences=args.sentences),
        batch_size=args.batch_size,
        shuffle=False)

    test_unseen_loader = torch.utils.data.DataLoader(
        AttributeDataset(args.data_dir, args.dataset,
                         features_path=args.gan_path,
                         mode='test_unseen',
                         generalized=True,
                         normalize=args.normalize,
                         sentences=args.sentences),
        batch_size=args.batch_size,
        shuffle=False)

    # instanciate the models
    if args.mlp:
        mlp = MLP(args.dim_input, [args.nhidden*2], args.nhidden)
    else:
        mlp = LinearProjection(args.dim_input, args.nhidden)
    embed = LinearProjection(args.nhidden, args.dim_embed)

    if args.sentences:
        cam_key = 'sentences'
    else:
        cam_key = 'emb'

    if args.gan_path is not None:
        cam_key = 'full_' + cam_key

    cam = torch.from_numpy(test_seen_loader.dataset.data[cam_key].T)
    proxies = ProxyNet(args.n_classes, args.dim_embed, proxies=cam)

    model = Base(mlp, embed, proxies)

    criterion = ProxyLoss(temperature=args.temp)

    if args.cuda:
        mlp.cuda()
        embed.cuda()
        model.cuda()
        proxies.cuda()

    # loading
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    txt = ("=> loaded checkpoint '{}' (epoch {})"
           .format(args.model_path, checkpoint['epoch']))
    print(txt)

    compute_scores(test_seen_loader, test_unseen_loader,
                   model, criterion)


def get_features(loader, model, criterion, proxies):
    feats = []
    probs = []
    model.eval()
    with torch.no_grad():
        for i, (img, labels) in enumerate(loader):
            if args.cuda:
                img = img.cuda(non_blocking=True)

            # compute output
            output = model(img)
            feats.append(output)
            probs.append(criterion.classify(output, proxies).cpu())

    return np.vstack(feats), np.vstack(probs)


def compute_scores(seen_loader, unseen_loader, model, criterion):
    if args.sentences:
        key = 'sentences'
    else:
        key = 'emb'
    proxies = seen_loader.dataset.data['full_' + key].T
    proxies = torch.from_numpy(proxies).cuda()

    print('\nExtracting testing seen set...')
    feats_seen, probs_seen = get_features(
        seen_loader, model, criterion, proxies)

    print('Extracting testing unseen set...\n')
    feats_unseen, probs_unseen = get_features(
        unseen_loader, model, criterion, proxies)

    seen_idx = np.asarray([*seen_loader.dataset.data['classnames'].keys()])

    dirname = os.path.dirname(args.model_path)
    fname = os.path.join(dirname, 'feat.npz')

    gamma = calibrate(probs_seen, probs_unseen,
                      seen_loader.dataset.data['Y_orig'],
                      unseen_loader.dataset.data['Y_orig'],
                      seen_idx)

    np.savez(fname,
             feats_seen=feats_seen, probs_seen=probs_seen,
             feats_unseen=feats_unseen, probs_unseen=probs_unseen,
             emb_seen=seen_loader.dataset.data[key].T,
             emb_unseen=unseen_loader.dataset.data[key].T,
             y_seen=seen_loader.dataset.data['Y_orig'],
             y_unseen=unseen_loader.dataset.data['Y_orig'],
             mapping_seen=seen_loader.dataset.data['mapping'],
             mapping_unseen=unseen_loader.dataset.data['mapping'],
             emb_full=seen_loader.dataset.data['full_' + key].T,
             seen_idx=np.unique(seen_loader.dataset.data['Y_orig']),
             unseen_idx=np.unique(unseen_loader.dataset.data['Y_orig']),
             gamma=gamma)


if __name__ == '__main__':
    main()

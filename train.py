import argparse
import os
import json
import numpy as np
import time
from data import AttributeDataset
from models import LinearProjection, MLP, Base
from models import ProxyNet, ProxyLoss
from utils import AverageMeter, get_data_stats
from validate import calibrate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# Training settings
parser = argparse.ArgumentParser(description='PyTorch DML')
# hyper-parameters
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 1e-2)')
parser.add_argument('--seed', type=int, default=456, metavar='S',
                    help='random seed (default: 1)')
# flags
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='de-enables CUDA training')
parser.add_argument('--no-decay', action='store_true', default=False,
                    help='de-enables LR decay')
# model
parser.add_argument('--dim-input', type=int, default=2048, metavar='N',
                    help='input dimensions (default: 2048)')
parser.add_argument('--temp', type=float, default=1, metavar='LR',
                    help='softmax temperature (default: 1)')
parser.add_argument('--ent', type=float, default=0.1, metavar='LR',
                    help='amount of entropy reg (default: 0.1)')
parser.add_argument('--margin', type=float, default=0.2, metavar='LR',
                    help='margin in the entropy reg (default: 0.2)')
parser.add_argument('--validation', action='store_true', default=False,
                    help='Use validation sets, does not work with gan path')
parser.add_argument('--nhidden', type=int, default=1024,
                    help='number of hidden units')
parser.add_argument('--mlp', action='store_true', default=False,
                    help='MLP instead of linear projection (default: False)')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='normalize data between 0 and 1')
# plumbing
parser.add_argument('--dataset', type=str, required=True,
                    help='sun|cub|awa1')
parser.add_argument('--sentences', action='store_true', default=False,
                    help='Use sentences, only for CUB')
parser.add_argument('--gan-path', type=str, default=None,
                    help='Path to GAN features')
parser.add_argument('--data-dir', type=str, metavar='DD', default='data',
                    help='data folder path')
parser.add_argument('--exp-dir', type=str, default='exp', metavar='ED',
                    help='folder for saving exp')
parser.add_argument('--m', type=str, default=None, metavar='M',
                    help='message')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.lr_decay = not args.no_decay

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# basic stats
if args.validation:
    n_seen, n_val_unseen, n_unseen, n_attr = get_data_stats(
        args.dataset, args.validation)
else:
    n_seen, n_unseen, n_attr = get_data_stats(args.dataset)
args.n_classes = n_seen


if args.sentences:
    args.dim_embed = 1024
else:
    args.dim_embed = n_attr

# create experiment folder
basic = '%s' % (args.dataset)
if args.m is not None:
    basic = basic + '_' + args.m
path = os.path.join(args.exp_dir, basic)
if not os.path.exists(path):
    os.makedirs(path)

# saving logs
with open(os.path.join(path, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=4, sort_keys=True)


def write_logs(txt, logpath=os.path.join(path, 'logs.txt')):
    with open(logpath, 'a') as f:
        f.write('\n')
        f.write(txt)


def main():
    if args.gan_path is None:
        both = False
    else:
        both = True

    if args.validation:
        train_loader = torch.utils.data.DataLoader(
            AttributeDataset(args.data_dir, args.dataset,
                             features_path=args.gan_path,
                             mode='train',
                             both=both,
                             normalize=args.normalize,
                             sentences=args.sentences),
            batch_size=args.batch_size,
            shuffle=True)
        val_seen_loader = torch.utils.data.DataLoader(
            AttributeDataset(args.data_dir, args.dataset,
                             features_path=args.gan_path,
                             mode='val_seen',
                             generalized=True,
                             normalize=args.normalize,
                             sentences=args.sentences),
            batch_size=args.batch_size,
            shuffle=False)
        val_unseen_loader = torch.utils.data.DataLoader(
            AttributeDataset(args.data_dir, args.dataset,
                             features_path=args.gan_path,
                             mode='val_unseen',
                             generalized=True,
                             normalize=args.normalize,
                             sentences=args.sentences),
            batch_size=args.batch_size,
            shuffle=False)
    else:
        trainval_loader = torch.utils.data.DataLoader(
            AttributeDataset(args.data_dir, args.dataset,
                             features_path=args.gan_path,
                             mode='trainval',
                             both=both,
                             normalize=args.normalize,
                             sentences=args.sentences),
            batch_size=args.batch_size,
            shuffle=True)

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

    if args.validation:
        cam = torch.from_numpy(train_loader.dataset.data[cam_key].T)
    else:
        cam = torch.from_numpy(trainval_loader.dataset.data[cam_key].T)
    proxies = ProxyNet(args.n_classes, args.dim_embed, proxies=cam)

    model = Base(mlp, embed, proxies)

    criterion = ProxyLoss(temperature=args.temp)

    if args.cuda:
        mlp.cuda()
        embed.cuda()
        model.cuda()
        proxies.cuda()

    parameters_set = []

    layers = []
    for c in mlp.children():
        if isinstance(c, nn.Linear) or isinstance(c, nn.ModuleList):
            layers.extend(list(c.parameters()))

    for c in embed.children():
        if isinstance(c, nn.Linear):
            layers.extend(list(c.parameters()))

    parameters_set.append({'params': layers,
                           'lr': args.lr})

    optimizer = optim.SGD(
        parameters_set,
        lr=args.lr,
        momentum=0.9, nesterov=True,
        weight_decay=5e-5)

    n_parameters = sum([p.data.nelement()
                        for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    scheduler = CosineAnnealingLR(optimizer, args.epochs)

    best_acc = 0
    print('Random results:')
    if args.validation:
        validate(val_seen_loader, val_unseen_loader,
                 model, criterion)
    else:
        validate(test_seen_loader, test_unseen_loader,
                 model, criterion)

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        if args.lr_decay:
            scheduler.step()

        # train for one epoch
        if args.validation:
            train(train_loader, model, criterion, optimizer, epoch)
            validate(val_seen_loader, val_unseen_loader,
                     model, criterion)
        else:
            train(trainval_loader, model, criterion, optimizer, epoch)
            validate(test_seen_loader, test_unseen_loader,
                     model, criterion)

        # saving
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict()})

    print('\nFinal evaluation on last epoch model:')
    validate(test_seen_loader, test_unseen_loader, model, criterion)


def train(train_loader, model, criterion, optimizer, epoch):
    """Training loop for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    val_losses = AverageMeter()
    val_acc = AverageMeter()

    # switch to train mode
    model.train()
    proxies = model.proxy_net.proxies.weight
    end = time.time()

    for i, (img, l) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gan_path is None:
            labels = l
        else:
            labels, m_s, m_u = l

        if args.cuda:
            img = img.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            if args.gan_path is not None:
                m_s = m_s.cuda(non_blocking=True)
                m_u = m_u.cuda(non_blocking=True)

        # forward pass
        output = model(img)

        # compute losses
        loss, acc = criterion(output, labels, proxies)

        if args.gan_path is not None:
            probs = criterion.classify(output, proxies)
            seen_idx = train_loader.dataset.data['seen_Y']
            unseen_idx = train_loader.dataset.data['unseen_Y']

            entropy_seen = - (torch.log(probs[:, seen_idx]) * probs[:, seen_idx]).sum(1)
            entropy_unseen = - (torch.log(probs[:, unseen_idx]) * probs[:, unseen_idx]).sum(1)

            margin = args.margin

            seen_in = (entropy_seen * m_s) / (m_s.sum() + 1e-8)
            seen_out = (entropy_seen * m_u) / (m_u.sum() + 1e-8)
            seen_entropy = torch.relu(margin + seen_in.sum() - seen_out.sum())

            unseen_out = (entropy_unseen * m_s) / (m_s.sum() + 1e-8)
            unseen_in = (entropy_unseen * m_u) / (m_u.sum() + 1e-8)
            unseen_entropy = torch.relu(margin + unseen_in.sum() - unseen_out.sum())

            loss = loss + args.ent * (seen_entropy + unseen_entropy)

        val_losses.update(loss.item(), img.size(0))
        val_acc.update(acc, img.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    txt = ('\nEpoch [%02d]: Time %.3f\t Data %.3f\t'
           'Loss %.4f\t Acc %.4f' % \
           (epoch, batch_time.avg * i, data_time.avg * i,
            val_losses.avg, val_acc.avg * 100.))
    print(txt)
    write_logs(txt)


def extract(loader, model, criterion, proxies):

    proxies_s, proxies_u, proxies_f = proxies

    probs_s = []
    probs_u = []
    probs_f = []

    with torch.no_grad():
        for i, (img, labels) in enumerate(loader):
            if args.cuda:
                img = img.cuda(non_blocking=True)

            # compute output
            output = model(img)

            probs_s.append(criterion.classify(output, proxies_s).cpu())
            probs_u.append(criterion.classify(output, proxies_u).cpu())
            probs_f.append(criterion.classify(output, proxies_f).cpu())

    # numpy plumbery
    probs_s = np.vstack(probs_s)
    probs_u = np.vstack(probs_u)
    probs_f = np.vstack(probs_f)

    return [probs_s, probs_u, probs_f]


def validate(seen_loader, unseen_loader, model, criterion):
    # switch to evaluate mode
    model.eval()

    # proxies
    if args.sentences:
        key = 'sentences'
    else:
        key = 'emb'
    proxies_s = seen_loader.dataset.data[key].T
    proxies_u = unseen_loader.dataset.data[key].T

    key = 'full_' + key
    proxies_f = seen_loader.dataset.data[key].T

    proxies_s = torch.from_numpy(proxies_s).cuda()
    proxies_u = torch.from_numpy(proxies_u).cuda()
    proxies_f = torch.from_numpy(proxies_f).cuda()
    proxies = [proxies_s, proxies_u, proxies_f]

    probs_seen = extract(seen_loader, model, criterion, proxies)
    probs_unseen = extract(unseen_loader, model, criterion, proxies)

    seen_idx = np.asarray([*seen_loader.dataset.data['classnames'].keys()])

    calibrate(probs_seen[-1], probs_unseen[-1],
              seen_loader.dataset.data['Y_orig'],
              unseen_loader.dataset.data['Y_orig'],
              seen_idx, v=write_logs)


def save_checkpoint(state, folder=path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))


def to_numpy(x):
    return x.cpu().data.numpy()[0]


if __name__ == '__main__':
    main()

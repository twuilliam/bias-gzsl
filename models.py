import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import cosine_similarity


def compatibility(data, weights, cam):
    '''
    Computes x*W*A (compatibility function)
    '''
    projection = np.dot(np.dot(data, weights), cam)
    return projection


class Base(nn.Module):
    '''first try'''
    def __init__(self, mlp, embed, proxies=None):
        super(Base, self).__init__()
        self.mlp = mlp
        self.embed = embed
        self.relu = nn.ReLU()
        if proxies is None:
            pass
        else:
            self.proxy_net = proxies

    def forward(self, x):
        x = self.mlp(x)
        x = self.relu(x)
        x = self.embed(x)
        return x


class LinearProjection(nn.Module):
    '''Linear projection'''
    def __init__(self, n_in, n_out):
        super(LinearProjection, self).__init__()
        self.fc_embed = nn.Linear(n_in, n_out)

    def forward(self, x):
        x = self.fc_embed(x)
        return x


class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        ''' MLP

        Args:
            n_in: number of input units
            n_hidden: list of ints
                number of units in hidden layers
            n_out: number of output units
        '''
        super(MLP, self).__init__()

        units = [n_in] + n_hidden + [n_out]

        self.linear = nn.ModuleList([
            nn.Linear(n_in, n_out)
            for n_in, n_out in zip(units[:-1], units[1:])])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.n_layers = len(units) - 1

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.linear[i](x)
            if i < (self.n_layers - 1):
                x = self.relu(x)
                x = self.dropout(x)
        return x

    def extract_layers(self, x):
        out = []
        for i in range(self.n_layers):
            x = self.linear[i](x)
            if i < (self.n_layers - 1):
                x = self.relu(x)
                x = self.dropout(x)
            out.append(x)
        return out


class ProxyNet(nn.Module):
    """ProxyNet"""
    def __init__(self, n_classes, dim, proxies):
        super(ProxyNet, self).__init__()
        self.n_classes = n_classes
        self.dim = dim

        self.proxies = nn.Embedding(n_classes, dim,
                                    scale_grad_by_freq=False)

        self.proxies.weight = nn.Parameter(proxies, requires_grad=False)

    def forward(self, y_true):
        proxies_y_true = self.proxies(Variable(y_true))
        return proxies_y_true


class ProxyLoss(nn.Module):
    def __init__(self, temperature=1.):
        super(ProxyLoss, self).__init__()

        self.temperature = temperature
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, y, proxies):
        # positive distances and negative distances
        loss = self.softmax_embedding_loss(x, y, proxies)
        all_loss = loss.mean()

        preds = self.predict(x, proxies)
        acc = (y == preds).type(torch.FloatTensor).mean()

        return all_loss, acc

    def softmax_embedding_loss(self, x, y, proxies):
        idx = torch.from_numpy(np.arange(len(x), dtype=np.int)).cuda()
        diff_iZ = cosine_similarity(x, proxies)

        return self.loss(diff_iZ / self.temperature, y)

    def classify(self, x, proxies):
        idx = torch.from_numpy(np.arange(len(x), dtype=np.int)).cuda()
        diff_iZ = cosine_similarity(x, proxies)

        numerator_ip = torch.exp(diff_iZ[idx, :] / self.temperature)
        denominator_ip = torch.exp(diff_iZ / self.temperature).sum(1) + 1e-8

        probs = numerator_ip / denominator_ip[:, None]
        return probs

    def predict(self, x, proxies):
        probs = self.classify(x, proxies)
        return probs.max(1)[1].data

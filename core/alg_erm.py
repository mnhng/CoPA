import pandas as pd
import torch
import torch.nn.functional as func

from .util import copy_to, init_best, Batchify
from .score import validate


class AlgERM():
    def __init__(self, batch_size, steps, lr, weight_decay, **kwargs):
        self.batch_size = batch_size
        self.steps = steps
        self.opt_params = {'lr': lr, 'weight_decay': weight_decay}

    def training_round(self, net, opt, D_labeled):
        dev = next(net.parameters()).device

        stat = {}
        for i, minibatch in enumerate(D_labeled):
            minibatch = next(minibatch)
            ids = minibatch.pop('id')
            minibatch = copy_to(minibatch, dev)
            Y = minibatch.pop('Y')
            opt.zero_grad()
            error = func.cross_entropy(net(**minibatch), Y)
            error.backward()
            opt.step()

            stat[f'{i}th E'] = error.item()

        return stat

    def __call__(self, net, train_envs, valid_envs):
        opt = torch.optim.Adam(net.parameters(), **self.opt_params)
        D_labeled = [Batchify(E, self.batch_size) for E in train_envs]

        rows = []
        best = init_best(len(valid_envs))
        it = 0
        while it < self.steps:
            row = self.training_round(net, opt, D_labeled)
            it += len(train_envs)
            if (it//len(train_envs)) % 50 == 0:  # validating
                rows.append(row | validate(net, best, valid_envs, self.batch_size, it))

        print(self.__class__.__name__, [(i, f'{v:.4f}') for i, v in zip(best['iter'], best['risk_score'])])

        return best, pd.DataFrame(rows)

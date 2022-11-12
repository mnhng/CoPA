import pandas as pd
import torch
import torch.nn.functional as func

from .util import copy_to, init_best, Batchify
from .score import validate


def irm_penalty(logits, Y):
    scale = torch.tensor(1.).to(Y.device).requires_grad_()
    loss = func.cross_entropy(logits * scale, Y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


class AlgIRM():
    def __init__(self, batch_size, steps, lr, weight_decay, irm_lambda, **kwargs):
        self.batch_size = batch_size
        self.steps = steps
        self.opt_params = {'lr': lr, 'weight_decay': weight_decay}
        self.irm_lambda = irm_lambda

    def training_round(self, net, opt, D_labeled):
        dev = next(net.parameters()).device

        stat = {}
        for i, minibatch in enumerate(D_labeled):
            minibatch = next(minibatch)
            ids = minibatch.pop('id')
            minibatch = copy_to(minibatch, dev)
            Y = minibatch.pop('Y')

            opt.zero_grad()
            logits = net(**minibatch)
            error = func.cross_entropy(logits, Y)
            (error + self.irm_lambda * irm_penalty(logits, Y)).backward()
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

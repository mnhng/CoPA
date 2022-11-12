#!/usr/bin/env python
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import core


class GrayCNN(nn.Sequential):
    def __init__(self, hid, **kwargs):
        super().__init__(
            core.Lambda(lambda x: x.sum(dim=1, keepdim=True)),
            nn.Conv2d(1, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64, hid),
        )

    def forward(self, X, **kwargs):
        return super().forward(X)


class CNN(nn.Sequential):
    def __init__(self, hid, **kwargs):
        super().__init__(
            nn.Conv2d(2, 32, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64, hid),
        )

    def forward(self, X, **kwargs):
        return super().forward(X)


def get_trunk(trunk_name):
    if trunk_name == 'ora':
        return GrayCNN
    if trunk_name == 'cnn':
        return CNN
    raise ValueError(trunk_name)


def get_env(env_str):
    if env_str == 'c_cause':
        return core.CmnistDia
    if env_str == 'y_cause':
        return core.CmnistTriL
    if env_str == 'z_cause':
        return core.CmnistTriR


def run_experiment(args):
    args.out.mkdir(parents=True, exist_ok=True)
    print('Flags:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as fh:
        conf = json.load(fh)

    dataset = core.CmnistDataSplit('data/mnist', n_valid=5000, n_test=5000)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    nb_Y = 2
    setattr(args, 'nb_Z', 2)

    ptrain, pvalid, ptest = dataset.shuffle()

    d_train = [args.env(b, b, **ptrain)(n_tr) for b, n_tr in conf['train']]
    b, n_va = conf['valid']
    d_ext_val = args.env(b, b, **pvalid)(n_va)
    d_train, d_int_val = core.skim(d_train, len(d_ext_val))
    d_valid = [d_int_val, d_ext_val]
    d_test = [args.env(b, b, **ptest)(n_tt) for b, n_tt in conf['test']]

    print('Train Envs:', [len(d) for d in d_train])
    print('Valid Envs:', [len(d) for d in d_valid])
    print('Test  Envs:', [len(d) for d in d_test])

    rows = []
    for name in args.methods:
        assert len(name.split('_')) == 2, f'Must be MODEL_METHOD: {name=}'
        n_trunk, n_method = name.split('_')

        trunk = get_trunk(n_trunk)(args.hidden_dim)
        head = core.get_head(n_method)(args.hidden_dim, nb_Y, [('color', {'binary': 0})])
        model = core.Classifier(trunk, head).to(dev)

        best, _ = core.train_wrapper(n_method, model, d_train, d_valid, d_test, args)

        row = {'N': name}

        for val_method, weights in zip(['int', 'ext'], best['state_dict']):
            if weights is None:
                continue

            model.load_state_dict(weights)
            row |= core.risk_round(model, d_test, args.batch_size, val_method)

        rows.append(row)

    out = pd.DataFrame(rows)
    col_mask = [c for c in out.columns if c.endswith('f1')]
    col_mask = col_mask[len(d_train)+len(d_test)-1::len(d_train)+len(d_test)]
    print(out[['N'] + col_mask])
    out.to_csv(args.out/'result.csv', index=False)


if __name__ == '__main__':
    parser = core.get_parser('Colored MNIST')
    parser.add_argument('--config', '-f', type=pathlib.Path, required=True)
    parser.add_argument('--env', type=get_env, required=True)

    run_experiment(parser.parse_args())

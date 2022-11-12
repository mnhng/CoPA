#!/usr/bin/env python
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import core


class MLP(nn.Sequential):
    def __init__(self, hid_dim, in_dim, **kwargs):
        super().__init__(
            nn.Linear(in_dim, hid_dim),
        )

    def forward(self, X, **kwargs):
        return super().forward(X)


def get_env(env_str):
    if env_str == 'c_cause':
        return core.SynDia
    if env_str == 'y_cause':
        return core.SynTriL
    if env_str == 'z_cause':
        return core.SynTriR


def run_experiment(args):
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as fh:
        conf = json.load(fh)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    nb_Y, in_dim = 2, 2
    setattr(args, 'nb_Z', 2)

    W = torch.linalg.qr(torch.randn(2, 2))[0]

    d_train = [args.env(b, b, scramble=W)(n_tr) for b, n_tr in conf['train']]
    b, n_va = conf['valid']
    d_ext_val = args.env(b, b, scramble=W)(n_va)
    d_train, d_int_val = core.skim(d_train, len(d_ext_val))
    d_valid = [d_int_val, d_ext_val]
    d_test = [args.env(b, b, scramble=W)(n_tt) for b, n_tt in conf['test']]

    print('Train Envs:', [len(d) for d in d_train])
    print('Valid Envs:', [len(d) for d in d_valid])
    print('Test  Envs:', [len(d) for d in d_test])
    evaluator = core.RiskEvaluator(d_train, d_test, args.batch_size)

    rows = []
    for name in args.methods:
        assert len(name.split('_')) == 2, f'Must be MODEL_METHOD: {name=}'
        _, n_method = name.split('_')

        trunk = MLP(hid_dim=args.hidden_dim, in_dim=in_dim)
        head = core.get_head(n_method)(args.hidden_dim, nb_Y, [('color', {'binary': 0})])
        model = core.Classifier(trunk, head).to(dev)

        best, _ = core.train_wrapper(n_method, model, d_train, d_valid, d_test, args)

        row = {'N': name}

        for val_method, weights in zip(['int', 'ext'], best['state_dict']):
            if weights is None:
                continue

            model.load_state_dict(weights)
            row |= evaluator(model, val_method)

        rows.append(row)

    out = pd.DataFrame(rows)
    col_mask = [c for c in out.columns if c.endswith('f1')]
    col_mask = col_mask[len(d_train)+len(d_test)-1::len(d_train)+len(d_test)]
    print(out[['N'] + col_mask])
    out.to_csv(args.out/'result.csv', index=False)


if __name__ == '__main__':
    parser = core.get_parser('Synthetic Experiment')
    parser.add_argument('--config', '-f', type=pathlib.Path, required=True)
    parser.add_argument('--env', type=get_env, required=True)

    run_experiment(parser.parse_args())

#!/usr/bin/env python
import json
import pathlib

import numpy as np
import pandas as pd
import torch

import core


def run_experiment(args):
    args.out.mkdir(parents=True, exist_ok=True)
    print('Flags:')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as fh:
        conf = json.load(fh)

    img_dir = pathlib.Path(conf['img_dir'])

    metadata = pd.read_csv('data/ISIC/metadata.csv', low_memory=False)
    d_train = [core.ISIC_site(img_dir, metadata, c, args.zlabels) for c in conf['train']]
    d_ext_val = core.ISIC_site(img_dir, metadata, conf['valid'], args.zlabels)
    d_test = [core.ISIC_site(img_dir, metadata, c, args.zlabels) for c in conf['test']]

    print('finished data loading')

    d_train, d_int_val = core.skim(d_train, len(d_ext_val))
    d_valid = [d_int_val, d_ext_val]

    print('Train Envs:', [len(d) for d in d_train])
    print('Valid Envs:', [len(d) for d in d_valid])
    print('Test  Envs:', [len(d) for d in d_test])

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = []
    for name in args.methods:
        assert len(name.split('_')) == 2, f'Must be MODEL_METHOD: {name=}'
        n_trunk, n_method = name.split('_')

        trunk = core.get_trunk(n_trunk)(args.hidden_dim)
        Zs = [(zl, {'float': 0}) if zl in {'age_approx'} else (zl, core.ISIC_Z_DICT[zl]) for zl in args.zlabels]
        head = core.get_head(n_method)(args.hidden_dim, len(core.ISIC_Y), Zs)
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
    parser = core.get_parser('ISIC')
    parser.add_argument('--config', '-f', type=pathlib.Path, required=True)
    parser.add_argument('--zlabels', nargs='+', choices=['anatom_site_general', 'age_approx', 'sex'], required=True)

    run_experiment(parser.parse_args())

import argparse
import pathlib

import numpy as np
import torch


def copy_to(param_dict, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in param_dict.items()}


def init_best(nb_val_sets):
    return {
        'risk_score': [0]*nb_val_sets,
        'iter': [0]*nb_val_sets,
        'state_dict': [None]*nb_val_sets,
    }


def get_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--methods', nargs='+', type=str.lower, required=True)
    parser.add_argument('--hidden_dim', type=int, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--out', '-o', type=pathlib.Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--irm_lambda', type=float)

    return parser


def skim(datasets, ref_size):
    portion =  ref_size / sum(len(d) for d in datasets)
    val_data, train_data = [], []
    for d in datasets:
        v, t = torch.utils.data.random_split(d, [portion, 1-portion], generator=torch.Generator().manual_seed(42))
        val_data.append(v)
        setattr(t, 'E', d.E)
        train_data.append(t)

    return train_data, sum(val_data[1:], val_data[0])


class Batchify():
    def __init__(self, dataset, batch_size):
        self.data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.iter_ = self.data.__iter__()

    def __next__(self):
        try:
            return next(self.iter_)
        except StopIteration:
            self.iter_ = self.data.__iter__()
            return next(self.iter_)

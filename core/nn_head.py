import torch
import torch.nn as nn


def get_head(name):
    if name == 'copa':
        return CoPAHead

    return LinearHead


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class LinearHead(nn.Linear):
    def __init__(self, dim, nb_Y, nb_Z, **kwargs):
        super().__init__(dim, nb_Y, **kwargs)

    def forward(self, input, **kwargs):
        return super().forward(input)


class CoPAHead(nn.Linear):
    def __init__(self, dim, nb_Y, Zs, device=None, dtype=None):
        super().__init__(dim, nb_Y, False, device, dtype)
        self.aux = nn.Linear(sum([len(mapping) for _, mapping in Zs]), nb_Y)

    def _get_log_odd(self, input, Z):
        return nn.functional.log_softmax(super().forward(input) + self.aux(Z.float()), dim=-1)

    def forward(self, input, Z, prior='YZE', no_Z_sampling=False, **kwargs):
        if Z.dim() == 1:
            Z = Z[:, None]
        if no_Z_sampling:
            bs, nZ = Z.size(0), 5

            candidates = Z[torch.randperm(bs)[:nZ]]

            odd = self._get_log_odd(input, candidates[:1].expand(bs, -1)).exp()
            for i in range(1, nZ):
                odd += self._get_log_odd(input, candidates[i:i+1].expand(bs, -1)).exp()
            log_odd = (odd/nZ).log()

        else:
            log_odd = self._get_log_odd(input, Z)

        return log_odd if prior is None else log_odd + kwargs[prior].log()

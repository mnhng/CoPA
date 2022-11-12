import torch

from .prevalence import ext_YE, ext_YZE
from .dataset import QuadInMem


#         S
#        / \
#       Y   Z
#        \ /
#         X
class Diamond():
    def __init__(self, beta_y, beta_z, **kwargs):
        self.alpha = 0.3
        self.gamma = 0.5
        # beta_y = beta_z = 1 => perfectly correlated
        self.beta_y = beta_y
        self.beta_z = beta_z
        self._generator_init(**kwargs)
        self.label = f'b{beta_y:.2f}' if beta_y == beta_z else f'bY{beta_y:.2f} bZ{beta_z:.2f}'

    def _get_YZ(self, n):
        S = torch.rand(n)
        Y = (self.beta_y*S + (1-self.beta_y)*self.alpha > self.gamma).long()
        Z = (self.beta_z*S + (1-self.beta_z)*torch.rand(n) > .5).long()
        return Y, Z

    def __call__(self, n):
        Y, Z = self._get_YZ(n)
        ids, X = self._generate(Y, Z)

        nb_Y = 2
        eY, eZ = self._get_YZ(100000)
        YE = ext_YE(eY[:n], nb_Y)
        YZE = ext_YZE(eY[:n], nb_Y, eZ[:n], Z)

        return QuadInMem(X, Y, Z, YE, YZE, self.label, ids)


#       Y-->Z
#        \ /
#         X
class TriagL():
    def __init__(self, beta_y, beta_z, **kwargs):
        self.alpha = 0.3
        self.gamma = 0.5
        # beta_y = beta_z = 1 => perfectly correlated
        self.beta_y = beta_y
        self.beta_z = beta_z
        self._generator_init(**kwargs)
        self.label = f'b{beta_y:.2f}' if beta_y == beta_z else f'bY{beta_y:.2f} bZ{beta_z:.2f}'

    def _get_YZ(self, n):
        Y = (self.beta_y*torch.rand(n) + (1-self.beta_y)*self.alpha > self.gamma).long()
        Z = (self.beta_z*Y/2 + (1-self.beta_z/2)*torch.rand(n) > .5).long()
        return Y, Z

    def __call__(self, n):
        Y, Z = self._get_YZ(n)
        ids, X = self._generate(Y, Z)

        nb_Y = 2
        eY, eZ = self._get_YZ(100000)
        YE = ext_YE(eY[:n], nb_Y)
        YZE = ext_YZE(eY[:n], nb_Y, eZ[:n], Z)

        return QuadInMem(X, Y, Z, YE, YZE, self.label, ids)


#       Y<--Z
#        \ /
#         X
class TriagR():
    def __init__(self, beta_y, beta_z, **kwargs):
        self.alpha = 0.3
        self.gamma = 0.5
        # beta_y = beta_z = 1 => perfectly correlated
        self.beta_y = beta_y
        self.beta_z = beta_z
        self._generator_init(**kwargs)
        self.label = f'b{beta_y:.2f}' if beta_y == beta_z else f'bY{beta_y:.2f} bZ{beta_z:.2f}'

    def _get_YZ(self, n):
        Z = (torch.rand(n) > .5).long()
        Y = (self.beta_y*(Z+torch.rand(n))/2 + (1-self.beta_y)*self.alpha > self.gamma).long()
        return Y, Z

    def __call__(self, n):
        Y, Z = self._get_YZ(n)
        ids, X = self._generate(Y, Z)

        nb_Y = 2
        eY, eZ = self._get_YZ(100000)
        YE = ext_YE(eY[:n], nb_Y)
        YZE = ext_YZE(eY[:n], nb_Y, eZ[:n], Z)

        return QuadInMem(X, Y, Z, YE, YZE, self.label, ids)

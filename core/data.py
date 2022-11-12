import pathlib

import torch
import numpy as np

from .io import load_idx


def shuffle_(*arrays):
    idxs = np.random.permutation(len(arrays[0]))
    return [array[idxs] for array in arrays]


class SynGenerator():
    def _generator_init(self, scramble=None):
        self.mu_y = torch.tensor([-.1, .1])
        self.mu_z = torch.tensor([ -1,  1])
        self.std_y = 0.1
        self.std_z = 0.1
        self.scramble = scramble

    def _generate(self, Y, Z):
        c1 = torch.where(Y==1, self.mu_y[1], self.mu_y[0]) + torch.randn(Y.shape)*self.std_y
        c2 = torch.where(Z==1, self.mu_z[1], self.mu_z[0]) + torch.randn(Z.shape)*self.std_z

        X = torch.cat((c1.unsqueeze(1), c2.unsqueeze(1)), 1)
        if self.scramble is not None:
            X = X @ self.scramble

        ids = np.asarray([f'{a:.2f}{b:.2f}' for a, b in zip(c1, c2)])

        return ids, X


class CmnistGenerator():
    def _generator_init(self, ids, images, labels):
        self.ids = ids
        self.images = images.float() / 255.

        indices = torch.arange(len(labels), dtype=int)
        neg_mask = labels < 5
        self.pos = indices[~neg_mask]
        self.neg = indices[neg_mask]

    def _generate(self, Y, Z):
        indices = torch.where(Y==1, self.pos[:len(Y)], self.neg[:len(Y)])
        images = self.images[indices]
        ids = self.ids[indices]

        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.arange(len(images), dtype=int), Z, :, :] *= 0

        return ids, images


class CmnistDataSplit():
    def __init__(self, root, n_valid, n_test):
        IMG_PATH = pathlib.Path(root)/'train-images-idx3-ubyte.gz'
        LBL_PATH = pathlib.Path(root)/'train-labels-idx1-ubyte.gz'
        # Load MNIST
        self.imgs = torch.tensor(load_idx(IMG_PATH))
        self.lbls = torch.tensor(load_idx(LBL_PATH))
        self.ids = np.asarray([str(i) for i in range(len(self.imgs))])
        self.j = self.imgs.shape[0] - n_test
        self.i = self.j - n_valid
        assert self.i > 0 and self.j > 0

    def shuffle(self):
        imgs, lbls, ids = shuffle_(self.imgs, self.lbls, self.ids)
        train = {'images': imgs[:self.i], 'labels': lbls[:self.i], 'ids': ids[:self.i]}
        valid = {'images': imgs[self.i:self.j], 'labels': lbls[self.i:self.j], 'ids': ids[self.i:self.j]}
        test = {'images': imgs[self.j:], 'labels': lbls[self.j:], 'ids': ids[self.j:]}

        return train, valid, test

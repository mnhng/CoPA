import numpy as np
import torch
import torch.nn.functional as func
from sklearn.neural_network import MLPClassifier


def prevalence_(Y_vec, pseudo_count):
    counts = Y_vec.sum(dim=0) + pseudo_count
    return (counts / counts.sum())


def ext_YE(labels, nb_Y):
    assert len(labels.shape) == 1, labels.shape
    return prevalence_(func.one_hot(labels, nb_Y), nb_Y).repeat(len(labels), 1)


def model_YZE(Y, nb_Y, Z, Z_out=None):
    assert len(Y.shape) == 1, Y.shape

    if Z.dim() == 1:
        Z = Z.reshape(-1, 1)

    clf1 = MLPClassifier(hidden_layer_sizes=(20, 20, 20), solver='lbfgs')

    if Z_out is None:
        mid = len(Y)//2
        Z_out = Z

        clf1.fit(Z[:mid], Y[:mid])

        clf2 = MLPClassifier(hidden_layer_sizes=(20, 20, 20), solver='lbfgs')
        clf2.fit(Z[mid:], Y[mid:])

        out = [clf2.predict_proba(Z_out[:mid]), clf1.predict_proba(Z_out[mid:])]
        return np.concatenate(out, axis=0)
    else:
        clf1.fit(Z, Y)
        return clf1.predict_proba(Z_out)


def ext_YZE(labels, nb_Y, Z, Z_out=None):
    assert len(labels.shape) == 1, labels.shape
    Y_vec = func.one_hot(labels, nb_Y)

    if Z_out is None:
        Z_out = Z
    out = torch.empty((len(Z_out), nb_Y), dtype=torch.float32, device=Y_vec.device)
    for z_val in Z_out.unique():
        out[Z_out == z_val] = prevalence_(Y_vec[Z == z_val], nb_Y)

    return out


def LOO_YE(labels, nb_Y):
    assert len(labels.shape) == 1, labels.shape
    Y_vec = func.one_hot(labels, nb_Y)

    out = torch.empty_like(Y_vec, dtype=torch.float32)
    idx = torch.arange(len(Y_vec), dtype=torch.int)
    for i in range(len(Y_vec)):
        out[i] = prevalence_(Y_vec[idx != i], nb_Y)

    return out

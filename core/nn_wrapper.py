import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, featurizer, classifier):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier

    def phi(self, X, **params):
        return self.featurizer(X, **params)

    def forward(self, X, return_feat=False, **params):
        feat = self.featurizer(X, **params)
        if return_feat:
            return self.classifier(feat, **params), feat
        return self.classifier(feat, **params)

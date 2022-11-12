import numpy as np
import pandas as pd
import torch

from .dataset import QuadInMem
from .prevalence import LOO_YE, model_YZE


CXR_Z_DICT = {
    'Sex': {'F': 0, 'M': 1, 'U': 2},
    'AP/PA': {'AP': 0, 'PA': 1, 'LL': 2},
}


def CXR_site(root_dir, site, zlabels):
    if site == 'CXR8':
        X = torch.load(root_dir/'cxr8.pt')
        df = pd.read_csv(root_dir/'cxr8.csv')
    elif site == 'PadChest':
        X = torch.load(root_dir/'padchest.pt')
        df = pd.read_csv(root_dir/'padchest.csv')
    elif site == 'CheXpert':
        X = torch.load(root_dir/'cxpert.pt')
        df = pd.read_csv(root_dir/'cxpert.csv')
    else:
        raise ValueError(site)

    assert len(df) == len(X), f'{len(df)=} {len(X)=}'

    nb_Y = 2
    Y = torch.as_tensor(df['Pneumonia'].to_numpy()).long()

    Z = []
    for zl in zlabels:
        if zl in {'Age'}:
            comp = torch.tensor(df[zl].to_numpy())
            Z.append(comp[:, None])
        else:
            comp = torch.tensor(df[zl].map(CXR_Z_DICT[zl]).to_numpy())
            Z.append(torch.nn.functional.one_hot(comp, len(CXR_Z_DICT[zl])))
    Z = torch.cat(Z, dim=-1)

    YE = LOO_YE(Y, nb_Y)
    YZE = model_YZE(Y, nb_Y, Z)
    ids = df['Path'].to_numpy()

    return QuadInMem(X.expand(-1, 3, -1, -1), Y, Z, YE, YZE, site, ids)

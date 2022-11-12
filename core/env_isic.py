import numpy as np
import pandas as pd
import torch

from .dataset import QuadInMem
from .prevalence import LOO_YE, model_YZE

ISIC_Y = {  # NOTE: no mapping for NaN
    'benign': 0,
    'malignant': 1
}

ISIC_Z_DICT = {
    'anatom_site_general': {
        'anterior torso': 0,
        'head/neck': 1,
        'lateral torso': 2,
        'lower extremity': 3,
        'oral/genital': 4,
        'palms/soles': 5,
        'posterior torso': 6,
        'upper extremity': 7,
        float('nan'): 8,
    },
    'sex': {'female': 0, 'male': 1, float('nan'): 2},
    # image_type
    'diagnosis_confirm_type': {
        'confocal microscopy with consensus dermoscopy': 0,
        'histopathology': 1,
        'serial imaging showing no change': 2,
        'single image expert consensus': 3,
        float('nan'): 4,
    },
    'dermoscopic_type': {
        'contact non-polarized': 0,
        'contact polarized': 1,
        'non-contact polarized': 2,
        float('nan'): 3,
    },
    'image_type': {
        'dermoscopic': 0,
        'overview': 1,
        float('nan'): 2,
    },
    'age_approx': {
        0: 0, 5: 0,
        10: 1, 15: 1,
        20: 2, 25: 2,
        30: 3, 35: 3,
        40: 4, 45: 4,
        50: 5, 55: 5,
        60: 6, 65: 6,
        70: 7, 75: 7,
        80: 8, 85: 8,
        90: 9, 95: 9,
        float('nan'): 10,
    },
}


def ISIC_site(img_dir, metadata, site, zlabels):
    cache = torch.load(img_dir/f'{site}.pt')
    metadata = metadata[metadata['site'] == site]
    assert len(metadata) == len(cache), f'{len(metadata)=} {len(cache)=}'

    mask = metadata['benign_malignant'].isin(ISIC_Y)

    X = cache[np.arange(len(metadata))[mask]]
    metadata = metadata[mask]

    Y = torch.as_tensor(metadata['benign_malignant'].map(ISIC_Y).to_numpy()).long()

    Z = []
    for zl in zlabels:
        comp = torch.tensor(metadata[zl].map(ISIC_Z_DICT[zl]).to_numpy())
        if zl in {'age_approx'}:
            Z.append(comp[:, None])
        else:
            Z.append(torch.nn.functional.one_hot(comp, len(ISIC_Z_DICT[zl])))
    Z = torch.cat(Z, dim=-1)

    YE = LOO_YE(Y, len(ISIC_Y))
    YZE = model_YZE(Y, len(ISIC_Y), Z)
    ids = metadata['isic_id'].to_numpy()

    return QuadInMem(X, Y, Z, YE, YZE, site, ids)

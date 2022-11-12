import pathlib

import pandas as pd
import torch
import torchvision


def center_crop(paths, size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(size),
    ])

    images = []
    for fp in paths:
        images.append(transform(torchvision.io.read_image(fp)).unsqueeze(dim=0))

    return torch.cat(images, dim=0).float() / 255.


if __name__ == '__main__':
    root = pathlib.Path('.')

    frame = pd.read_csv('metadata.csv', usecols=['isic_id', 'site'])

    paths = frame['isic_id'].apply(lambda x: str(root/'raw'/f'{x}.JPG'))

    imgs = center_crop(paths, size=224)

    assert len(frame) == len(imgs)

    out_folder = root/'images'

    out_folder.mkdir(parents=True, exist_ok=True)
    for site in frame['site'].unique():
        torch.save(imgs[frame['site'] == site], out_folder/f'{site}.pt')

import torch
import torchvision


class QuadInMem(torch.utils.data.Dataset):
    def __init__(self, X, Y, Z, YE, YZE, label, ids):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.YE = YE
        self.YZE = YZE
        self.ids = ids
        self.E = label

    def __getitem__(self, index):
        return {
                'id': self.ids[index],
                'X': self.X[index], 'Y': self.Y[index], 'Z': self.Z[index],
                'YE': self.YE[index], 'YZE': self.YZE[index],
                }

    def __len__(self):
        return len(self.X)


class QuadOnDisk(torch.utils.data.Dataset):
    def __init__(self, X_path, Y, Z, YE, YZE, label, ids):
        self.X_path = X_path
        self.Y = Y
        self.Z = Z
        self.YE = YE
        self.YZE = YZE
        self.ids = ids
        self.E = label
        self.aug_fn = torchvision.transforms.RandomCrop(224)

    def __getitem__(self, index):
        img = torchvision.io.read_image(self.X_path[index]).float() / 255.
        return {
                'id': self.ids[index],
                'X': self.aug_fn(img), 'Y': self.Y[index], 'Z': self.Z[index],
                'YE': self.YE[index], 'YZE': self.YZE[index],
                }

    def __len__(self):
        return len(self.X_path)

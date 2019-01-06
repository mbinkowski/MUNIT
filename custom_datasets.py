from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image
from glob import glob

from torchvision.datasets import *

AVAILABLE = ['MNIST', 'SVHN', 'QUICKDRAW', 'CELEBA_QD', 'PORTRAIT', 'CELEBA_DRIT']

def get_default_transform(size): 
    return transforms.Compose([ 
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    ])

class QUICKDRAW(Dataset):
    def __init__(self, root='./', classes=['face'], transform=None, **kwargs):
        self.root = root
        data = [np.load(os.path.join(root, c + '.npy')) for c in classes]
        data = [d.reshape(d.shape[0], 28, 28, 1) for d in data]
        self.inputs = np.concatenate(data, axis=0)
        self.transform = transforms.Compose([transforms.ToPILImage(), transform])

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.transform(self.inputs[idx])


class MNIST(datasets.MNIST):
    def __init__(self, root='.', download=True, transform=None, train=True, **kwargs):
        super(MNIST, self).__init__(root=root, download=download, transform=transform, train=train)

    def __getitem__(self, index):
        return super(MNIST, self).__getitem__(index)[0]

class SVHN(datasets.SVHN):
    def __init__(self, root='.', download=True, transform=None, train=True, **kwargs):
        super(SVHN, self).__init__(root=root, download=download, transform=transform,
                                   split='train' if train else 'test')

    def __getitem__(self, index):
        return super(SVHN, self).__getitem__(index)[0]



class ImageFolder(Dataset):
    default_transforms = []

    def __init__(self, root='.', transform=None, **kwargs):
        self.files = glob(os.path.join(root, '*.jpg'))
        assert len(self.files) > 0, 'No jpg file found in the root directory %s.' % root
        print('Dataset initializer; found %d .jpg images in the root directory %s' % (len(self.files), root))
        self.transform = transforms.Compose(self.default_transforms + [transform] * (transform is not None))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        count = 0
        while True:
            try:
                image = Image.open(self.files[idx])
                return self.transform(image)
            except Exception as e:
                idx = np.random.randint(self.__len__())
                count += 1
                print(count, e)


class CELEBA(ImageFolder):
    default_transforms = [
        transforms.RandomApply((transforms.CenterCrop(144),), .9),
#        transforms.CenterCrop(160),
#        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(256, scale=(.75, 1), ratio=(1,1)),
#        transforms.RandomCrop(144),
        transforms.RandomHorizontalFlip()
    ]

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = transforms.functional.crop(image, 32, 0, 178, 178)
        return self.transform(image)


class CELEBA_QD(CELEBA):
    pass

class PORTRAIT(ImageFolder):
    default_transforms = [
        transforms.CenterCrop(178),
        transforms.Resize(256, Image.BICUBIC),
        transforms.RandomResizedCrop(256, scale=(.9, 1), ratio=(1,1)),
        transforms.RandomHorizontalFlip()
    ]

class CELEBA_DRIT(ImageFolder):
    default_transforms = [
        transforms.CenterCrop(160),
        transforms.Resize(256, Image.BICUBIC),
        transforms.RandomResizedCrop(256, scale=(.9, 1), ratio=(1,1)),
        transforms.RandomHorizontalFlip()
    ]

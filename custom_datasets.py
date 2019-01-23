from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image
from glob import glob

from torchvision.datasets import *

AVAILABLE = ['MNIST', 'SRMNIST', 'SMALLMNIST', 'SVHN', 'QUICKDRAW', 'CELEBA_QD', 'PORTRAIT', 'CELEBA_DRIT']

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

#    def __getitem__(self, index):
#        return super(MNIST, self).__getitem__(index)[0]

class SMALLMNIST(MNIST):
    def __init__(self, root='.', download=True, transform=None, train=True, **kwargs):
        trans = transforms.Pad((5,10,15,10), padding_mode='edge')
        if transform is not None:
            trans = transforms.Compose([trans, transform])
        super(SMALLMNIST, self).__init__(root=root, download=download,
                                         transform=trans, train=train, **kwargs)


class SRMNIST(MNIST):
    def __init__(self, root='.', download=True, transform=None, train=True, **kwargs):
        trans = transforms.Compose([
            transforms.Pad((7,7,7,7), padding_mode='edge'),
            transforms.RandomRotation(15, resample=Image.BILINEAR),
            transforms.RandomCrop(33)
           # transforms.RandomResizedCrop(28, scale=(.7, 1.), ratio=(1., 1.))
        ])
        if transform is not None:
            trans = transforms.Compose([trans, transform])
        super(SRMNIST, self).__init__(root=root, download=download,
                                         transform=trans, train=train, **kwargs)


class SVHN(datasets.SVHN):
    def __init__(self, root='.', download=True, transform=None, train=True, **kwargs):
        super(SVHN, self).__init__(root=root, download=download, transform=transform,
                                   split='train' if train else 'test')

#    def __getitem__(self, index):
#        return super(SVHN, self).__getitem__(index)[0]



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

class Folder(Dataset):
    default_transforms = []

    def __init__(self, root='.', transform=None, with_saving=False, suffix='', **kwargs):
        self.count = 0
        name = root.replace('data1', 'tmp1/binkowsm') 
        if name[-1] == '/':
            name = name[:-1]
        name += suffix
        if with_saving:
            if not os.path.exists(name):
                os.makedirs(name)
            if os.path.exists(name + '.pt'):
                self.ims = torch.load(name + '.pt').cpu()#.data.numpy()
                return
        self.files = glob(os.path.join(root, '*.jpg'))
        assert len(self.files) > 0, 'No jpg file found in the root directory %s.' % root
        print('Dataset initializer; found %d .jpg images in the root directory %s' % (len(self.files), root))
        self.transform = transforms.Compose(self.default_transforms  + [transform] * (transform is not None))
        if with_saving:
            self._save(name)

    def __len__(self):
        if hasattr(self, 'ims'):
            return len(self.ims)
        return len(self.files)

    def __getitem__(self, idx):
#        print('DataFolder sampling count', self.count)
#        self.count += 1
        if hasattr(self, 'ims'):
            return self.ims[idx], 0
        image = Image.open(self.files[idx])
        return self.transform(image), 0

    def _save(self, name):
        ims = []
        for i in range(self.__len__()):
            ims.append(self.__getitem__(i)[0].cpu())
            if i % 1000 == 0:
                print('Image Folder saving. Processed: %d' % i)
        self.ims = torch.stack(ims)
#        self.tensor = cuda(self.tensor)
        torch.save(torch.Tensor(self.ims), name + '.pt')


class EDGES(Dataset):
    left, name = 0, 'EDGES'
    def __init__(self, root='.', transform=None, with_saving=True, **kwargs):
        assert 'shoes' in root, root
        crop = lambda x: transforms.functional.crop(x, 0, self.left, 256, 256)
        t = transforms.Lambda(crop)
        if transforms is not None:
            t = transforms.Compose([t, transform])
        self.shoes = Folder(root=root, transform=t, with_saving=with_saving, 
                                 suffix=self.name.lower(), **kwargs)
        self.handbags = Folder(root=root.replace('shoes', 'handbags'),
                                    transform=t, with_saving=with_saving, 
                                    suffix=self.name.lower(), **kwargs)
        print('%s dataset: %d shoes, %d handbags' % (self.name, len(self.shoes), len(self.handbags)))
        self.count = 0

    def __len__(self):
        return len(self.shoes) + len(self.handbags)

    def __getitem__(self, idx):
#        print(self.name, 'samplied', self.count)
#        self.count += 1
        if idx < len(self.shoes):
            return self.shoes[idx][0], 0
        else:
            return self.handbags[idx - len(self.shoes)][0], 1

    def get_target_classes(self):
        return [0] * len(self.shoes) + [1] * len(self.handbags)

class SHOESHANDBAGS(EDGES):
    left, name = 256, 'SHOESHANDBAGS'


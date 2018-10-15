# -*- coding: utf-8 -*-
# Author: Ji Yang <jiyang.py@gmail.com>
# License: MIT

import random

import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

padding = transforms.Compose([transforms.Resize(202),
                              transforms.Pad(30, padding_mode='reflect'),
                              transforms.RandomRotation((-8, 8)),
                              transforms.RandomApply([transforms.RandomAffine(0, shear=8)]),
                              transforms.RandomCrop(256)])

strong_augmentation_transform = transforms.Compose([transforms.RandomChoice([padding]),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomApply([transforms.ColorJitter(brightness=0.1,
                                                                                                   contrast=0.1,
                                                                                                   saturation=0.1,
                                                                                                   hue=0.1)]),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])
                                                    ])

basic_augmentation_transform = transforms.Compose([transforms.RandomChoice([padding]),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.ToTensor()
                                                   ])

val_test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])
                                         ])


class SaltDataset(Dataset):
    def __init__(self, image, mask=None, transform=strong_augmentation_transform, is_train=True):
        self.image = image
        self.mask = mask
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        image = Image.fromarray(np.uint8(self.image[idx] * 255))
        seed = random.randint(6, 6 ** 6)

        random.seed(seed)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = val_test_transform(image)

        if self.is_train:
            mask = Image.fromarray(np.uint8(self.mask[idx]))
            random.seed(seed)
            if self.transform is not None:
                mask = self.transform(mask)
            else:
                mask = val_test_transform(mask)
            mask = (mask > 0.5).float()  # round resize artifact
            return {'image': image, 'mask': mask}

        return {'image': image}

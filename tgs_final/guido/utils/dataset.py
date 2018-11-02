#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

import cv2

from torch.utils import data
from torchvision import transforms

# image params
img_size_ori = 101
pad_size = 11  # 11, 101 * 2 + 11 * 2 = 224,  27, 101 * 2 + 27 * 2 = 256


def pad(base, mask=None):
    base = cv2.copyMakeBorder(base, 13, 14, 13, 14, cv2.BORDER_REFLECT_101)
    if mask is not None:
        mask = cv2.copyMakeBorder(mask, 13, 14, 13, 14, cv2.BORDER_REFLECT_101)
        return base, mask
    return base


def unpad(imgs):
    return imgs[:, 13:114, 13:114]


def puzzle_pad(base, mask=None):
    base = np.concatenate((base, base))
    base = np.concatenate((base, base), axis=1)
    base = cv2.copyMakeBorder(base, pad_size, pad_size, pad_size, pad_size,
                              cv2.BORDER_REFLECT_101)
    if mask is not None:
        mask = np.concatenate((mask, mask))
        mask = np.concatenate((mask, mask), axis=1)
        mask = cv2.copyMakeBorder(mask, pad_size, pad_size, pad_size, pad_size,
                                  cv2.BORDER_REFLECT_101)
        return base, mask
    return base


def un_puzzle_pad(imgs):
    result = np.zeros((4, imgs.shape[0], img_size_ori, img_size_ori))
    k = 0
    for i in range(2):
        for j in range(2):
            y1 = pad_size + (i * img_size_ori)
            y2 = pad_size + ((i + 1) * img_size_ori)
            x1 = pad_size + (j * img_size_ori)
            x2 = pad_size + ((j + 1) * img_size_ori)
            result[k] = imgs[:, y1:y2, x1:x2]
            k += 1
    return result.mean(axis=0)


def resize224(base, mask=None):
    base = cv2.resize(base, (224, 224))
    if mask is not None:
        mask = cv2.resize(mask, (224, 224))
        return base, mask
    return base


def resize128(base, mask=None):
    base = cv2.resize(base, (128, 128))
    if mask is not None:
        mask = cv2.resize(mask, (128, 128))
        return base, mask
    return base


def unresize(imgs):
    result = np.array([cv2.resize(img, (101, 101)) for img in imgs])
    return result


class TGSSaltDataset(data.Dataset):
    def __init__(self, data_dir, file_list, transform=None, mode='train'):
        self.base_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.file_list = file_list
        self.transform = transform
        self.mode = mode

    def to_tensor(self, base, mask=None):
        # To Tensor
        totensor = transforms.ToTensor()
        base = totensor(base)
        # Normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        base = normalize(base)
        if mask is not None:
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            mask = totensor(mask)
            return base, mask
        return base

    # def add_depth_channels(self, base):
    #     h, w, _ = base.shape
    #     print(type(base))
    #     for row, const in enumerate(np.linspace(0, 1, h)):
    #         base[row, :, 1] = const
    #         print(row)
    #         print(const)
    #         print(base[row, :, 1])
    #         if row == 11:
    #             exit()
    #     base[:, :, 2] = base[:, :, 0] * base[:, :, 1]
    #     return base

    def get_train_item(self, index):
        # numpy shape: (H, W, C)
        # torch shape: (C, H, W), ToTensor or torch.from_numpy
        file_id = self.file_list[index]

        base_path = os.path.join(self.base_dir, file_id + ".png")
        mask_path = os.path.join(self.mask_dir, file_id + ".png")

        base = cv2.imread(base_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # base, mask = puzzle_pad(base, mask)  # not resize224(base, mask)
        # base, mask = pad(base, mask)
        base, mask = resize128(base, mask)

        if self.transform is not None:
            base, mask = self.transform(base, mask)

        base, mask = self.to_tensor(base, mask)

        cover = (mask > 0.5).sum()
        nempty = 1.
        if cover < 8:
            nempty = 0.
        return base, mask, nempty, file_id

    def get_test_item(self, index):
        file_id = self.file_list[index]

        base_path = os.path.join(self.base_dir, file_id + ".png")

        base = cv2.imread(base_path)

        # base = puzzle_pad(base)
        # base = pad(base)
        base = resize128(base)

        base = self.to_tensor(base)

        return base, file_id

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        if self.mode == 'train':
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def __len__(self):
        return len(self.file_list)

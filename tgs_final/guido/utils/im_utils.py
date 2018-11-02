#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random

import numpy as np

import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import utils


# https://github.com/asanakoy/kaggle_carvana_segmentation/blob/master/albu/src/transforms.py
class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None):
        for t in self.transforms:
            x, mask = t(x, mask)
        return x, mask


class ImageOnly:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x, mask=None):
        return self.trans(x), mask


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


class HorizontalFlip:
    def __init__(self, prob=.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class ShiftScaleRotate:
    def __init__(self,
                 shift_limit=0.0625,
                 scale_limit=0.1,
                 rotate_limit=45,
                 prob=0.5):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            height, width, channel = img.shape

            angle = random.uniform(-self.rotate_limit, self.rotate_limit)
            scale = random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
            dx = round(random.uniform(-self.shift_limit,
                                      self.shift_limit)) * width
            dy = round(random.uniform(-self.shift_limit,
                                      self.shift_limit)) * height

            cc = math.cos(angle / 180 * math.pi) * scale
            ss = math.sin(angle / 180 * math.pi) * scale
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height],
            ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array(
                [width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)
            img = cv2.warpPerspective(
                img,
                mat, (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpPerspective(
                    mask,
                    mat, (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)

        return img, mask


class RandomBrightness:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1.0 + self.limit * random.uniform(-1, 1)

            maxval = np.max(img[..., :3])
            dtype = img.dtype
            img[..., :3] = clip(alpha * img[..., :3], dtype, maxval)
        return img


class RandomHueSaturationValue:
    def __init__(self,
                 hue_shift_limit=(-10, 10),
                 sat_shift_limit=(-25, 25),
                 val_shift_limit=(-25, 25),
                 prob=0.5):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(self.hue_shift_limit[0],
                                          self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(self.sat_shift_limit[0],
                                          self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0],
                                          self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image


def transform(base, mask=None, prob=.5):
    return DualCompose([
        HorizontalFlip(prob=.5),
        ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.10, rotate_limit=10, prob=.75),
        ImageOnly(RandomBrightness()),
        ImageOnly(RandomHueSaturationValue())
    ])(base, mask)


def imshow(im, title=None):
    """
    :param im:
        tensor type (size: _, height, width)
    """
    _, height, width = im.shape
    if _ == 1:
        im = np.repeat(im, 3, axis=0)
    img = im.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()


def imshow_example(dataloader):
    # Get a batch of training data
    imgs, masks, indexs = next(iter(dataloader))

    imgs = utils.make_grid(imgs, padding=0)
    masks = utils.make_grid(masks, padding=0)
    examples = torch.cat((imgs, masks), 1)

    imshow(examples)


# -------
"""
def load_image(path, mask=False):
    # Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad,
                             cv2.BORDER_REFLECT_101)

    if mask:
        # Convert mask to 0 and 1 format
        img = img[:, :, 0:1] // 255
    else:
        img = img / 255.0

    # img shape (height, width, _) to (_, height, width)
    print(img.shape)
    img = torch.from_numpy(img).float().permute([2, 0, 1])
    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
"""

# def random_brightness(image, limit=(-0.3, 0.3), u=0.5):
#     if random.random() < u:
#         alpha = 1.0 + random.uniform(limit[0], limit[1])
#         image = alpha * image
#         image = np.clip(image, 0., 1.)
#     return image

# def random_contrast(image, limit=(-0.3, 0.3), u=0.5):
#     if random.random() < u:
#         alpha = 1.0 + random.uniform(limit[0], limit[1])
#         coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
#         gray = image * coef
#         gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
#         image = alpha * image + gray
#         image = np.clip(image, 0., 1.)
#     return image

# def random_saturation(image, limit=(-0.3, 0.3), u=0.5):
#     if random.random() < u:
#         alpha = 1.0 + random.uniform(limit[0], limit[1])
#         coef = np.array([[[0.114, 0.587, 0.299]]])
#         gray = image * coef
#         gray = np.sum(gray, axis=2, keepdims=True)
#         image = alpha * image + (1.0 - alpha) * gray
#         image = np.clip(image, 0., 1.)
#     return image

# def random_gray(image, u=0.5):
#     if random.random() < u:
#         coef = np.array([[[0.114, 0.587, 0.299]]])
#         gray = np.sum(image * coef, axis=2)
#         image = np.dstack((gray, gray, gray))
#     return image

# def padding_augmentation(images):
#     # color = [0, 0, 0]
#     for i, image in enumerate(images):
#         images[i] = cv2.copyMakeBorder(image, y_min_pad, y_max_pad, x_min_pad,
#                                        x_max_pad, cv2.BORDER_REFLECT_101)
#         # images[i] = cv2.copyMakeBorder(
#         #     image,
#         #     y_min_pad,
#         #     y_max_pad,
#         #     x_min_pad,
#         #     x_max_pad,
#         #     cv2.BORDER_CONSTANT,
#         #     value=color)
#     return images

# def positional_augmentation(images):
#     row, col, _ = images[0].shape
#     # Random center crop and resize to original size
#     size_range = 10
#     size = random.randint(0, size_range)
#     h = row - 2 * size
#     w = col - 2 * size
#     for i, image in enumerate(images):
#         image = image[size:size + h, size:size + w]
#         images[i] = cv2.resize(image, (img_size_target, img_size_target))
#     # Horizontal Flip
#     if random.random() < 0.5:
#         for i, image in enumerate(images):
#             images[i] = cv2.flip(image, 1)
#     # Vertical Flip
#     # if random.random() < 0.5:
#     #     for i, image in enumerate(images):
#     #         images[i] = cv2.flip(image, 0)
#     # Rotation
#     degree = 10
#     angle = random.uniform(-degree, degree)
#     print(angle)
#     Rot_M = cv2.getRotationMatrix2D((col / 2, row / 2), angle, 1)
#     for i, image in enumerate(images):
#         images[i] = cv2.warpAffine(image, Rot_M, (col, row))
#     # Shear
#     # shear_range = 10
#     # pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
#     # pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
#     # pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
#     # pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
#     # shear_M = cv2.getAffineTransform(pts1, pts2)
#     # for i, image in enumerate(images):
#     #     images[i] = cv2.warpAffine(image, shear_M, (col, row))

#     # Add more translation/scale augmentations here...
#     return images

# def color_augmentation(image):
#     # Only applied to the base image, and not the mask layers.
#     # Color Jitter
#     # image = random_brightness(image, limit=(-0.08, 0.08), u=0.5)  # -0.5, 0.5
#     # image = random_contrast(image, limit=(-0.5, 0.5), u=0.5)
#     # image = random_saturation(image, limit=(-0.3, 0.3), u=0.5)
#     # image = random_gray(image, u=0.25)
#     # Add more color augmentations here...
#     return image

# def valid_transform(base, mask):
#     base, mask = padding_augmentation([base, mask])

#     # To Tensor
#     mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
#     aug = transforms.ToTensor()
#     base = aug(base)
#     mask = aug(mask)
#     # Normalize
#     aug = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     base = aug(base)

#     return base, mask

# def test_transform(base):
#     base = padding_augmentation([base])[0]

#     # To Tensor
#     aug = transforms.ToTensor()
#     base = aug(base)
#     # Normalize
#     aug = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     base = aug(base)

#     return base

# def test_transform2(base):
#     base = padding_augmentation([base])[0]
#     base = cv2.flip(base, 1)

#     # To Tensor
#     aug = transforms.ToTensor()
#     base = aug(base)
#     # Normalize
#     aug = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     base = aug(base)

#     return base

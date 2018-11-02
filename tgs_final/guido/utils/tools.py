#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x.data[0]  # Variable(x)
        return x[0]

    return x.item()


def save_state(model, epoch, max_iout, model_path):
    torch.save({
        "model": model.state_dict(),
        "max_iout": max_iout,
        "epoch": epoch,
    }, str(model_path))


def load_state(model, model_path):
    if os.path.exists(model_path):
        state = torch.load(model_path)
        start_epoch = state["epoch"]
        max_iout = state["max_iout"]
        model.load_state_dict(state["model"])
        print(f"Restore model, epoch: {start_epoch}, max_iout: {max_iout:.4f}")
        return model, start_epoch, max_iout
    else:
        return model, 1, 0.


def filter_image(img):
    # https://www.kaggle.com/divrikwicky/u-net-with-simple-resnet-blocks-forked/notebook
    # https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/64875
    # excluded all images that were below 1.5% of salt coverage
    if img.sum() < 30:  # 102, 101 * 101 * 0.015 = 153.015
        return np.zeros(img.shape)
    else:
        return img


def rle_encode(im):
    """
    Converting the decoded image to rle mask

    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# https://www.kaggle.com/shaojiaxin/u-net-resnet-v3-stratifiedkfold
def get_mask_type(mask):
    border = 10
    outer = np.zeros((101 - 2 * border, 101 - 2 * border), np.float32)
    outer = cv2.copyMakeBorder(
        outer,
        border,
        border,
        border,
        border,
        borderType=cv2.BORDER_CONSTANT,
        value=1)

    cover = (mask > 0.5).sum()
    if cover < 8:
        return 0  # empty
    if cover == ((mask * outer) > 0.5).sum():
        return 1  # border
    if np.all(mask == mask[0]):
        return 2  # vertical

    percentage = cover / (101 * 101)
    if percentage < 0.15:
        return 3
    elif percentage < 0.25:
        return 4
    elif percentage < 0.50:
        return 5
    elif percentage < 0.75:
        return 6
    else:
        return 7


def histcoverage(coverage):
    histall = np.zeros(8)
    for c in coverage:
        histall[c] += 1
    return histall


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i


"""
from skimage.color import gray2rgb
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels

def crf(original_image, mask_img):
    # Converting annotated image to RGB if it is Gray scale
    if (len(mask_img.shape) < 3):
        mask_img = gray2rgb(mask_img)

    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (
        mask_img[:, :, 2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2

    # Setting up the CRF model
    d = densecrf.DenseCRF2D(original_image.shape[1], original_image.shape[0],
                            n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=densecrf.DIAG_KERNEL,
        normalization=densecrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 10 steps
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0], original_image.shape[1]))
"""

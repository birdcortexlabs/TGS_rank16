#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import os
from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt
# from skimage.io import imread

# from utils.loss import iou_metric_batch, iou_metric_batch2, get_iou_vector,
from utils.loss import do_kaggle_metric

# from utils.tools import filter_image
# from utils.tools import crf

train_dir = "input/train"


def get_best_threshold(valid_preds, valid_masks, ids_valid):
    iout_by_threshold = []
    for threshold in tqdm(np.linspace(0.3, 0.7, 31)):
        # Reverse sigmoid function: Use code below because the sigmoid activation was removed
        th = threshold
        threshold = np.log(threshold / (1 - threshold))
        if th == 0.45:
            print(threshold)
        valid_bin_preds = np.int32(valid_preds > threshold)

        # crf_valid_bin_preds = []
        # for i, p_mask in enumerate(valid_bin_preds):
        #     ori_img = imread(
        #         os.path.join(train_dir, f"images/{ids_valid[i]}.png"))
        #     crf_output = crf(ori_img, p_mask)
        #     crf_valid_bin_preds.append(crf_output)

        # valid_bin_preds = np.array(crf_valid_bin_preds)

        # iou1 = iou_metric_batch(valid_masks, filter_image(valid_bin_preds))
        # iou2 = iou_metric_batch2(valid_masks, filter_image(valid_bin_preds))
        # iou3 = get_iou_vector(filter_image(valid_bin_preds), valid_masks)
        iout = do_kaggle_metric(valid_bin_preds, valid_masks)
        iout_by_threshold.append((iout, threshold))
    best_iout, best_threshold = max(iout_by_threshold)
    print(f'best_iout: {best_iout:.4f}, best_threshold: {best_threshold:.2f}')

    # plot_gen_threshold(best_threshold, best_iou, iou_by_threshold)

    return best_threshold


def plot_gen_threshold(best_threshold, best_iou, iou_by_threshold):
    iou_by_threshold = np.array(iou_by_threshold)
    plt.plot(iou_by_threshold[:, 1], iou_by_threshold[:, 0])
    plt.plot(best_threshold, best_iou, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title(f"Best IoU {best_iou:.4f} by Threshold {best_threshold:.2f}")
    plt.legend()
    plt.show()

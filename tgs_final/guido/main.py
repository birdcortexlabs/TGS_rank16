#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib
from PIL import Image

import torch
from torch.backends import cudnn

from utils import train
# from utils.dataset import img_size_ori
from utils.eval import infer, submit_result
# from utils.im_utils import imshow_example
from utils.tools import get_mask_type

matplotlib.use('TkAgg')

# data params
train_path = "input/train.csv"
depths_path = "input/depths.csv"
sample_path = "input/sample_submission.csv"

data_dir = "input"
train_dir = "input/train"
test_dir = "input/test"
model_dir = "model"

# model params
batch_size = 12

print('@%s:  ' % os.path.basename(__file__))

if True:
    SEED = 235202
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    print(f'\tset random seed={SEED}')

use_gpu = torch.cuda.is_available()
if use_gpu:
    # inbuilt cudnn auto-tuner to find the fastest convolution algorithms
    cudnn.benchmark = True
    print('\tset cuda environment, use gpu')


def main(n_fold=-1, phase=None):
    # Loading of training/testing ids and depths
    train_df = pd.read_csv(train_path, index_col="id", usecols=[0])
    depths_df = pd.read_csv(depths_path, index_col="id")
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    print(f"train_df:\n{train_df.head()}")
    print(f"test_df:\n{test_df.head()}")

    train_df["masks"] = [
        np.array(
            Image.open(os.path.join(train_dir,
                                    f"masks/{idx}.png")).convert('L')) / 255
        for idx in train_df.index
    ]

    # train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    # train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    train_df["coverage_class"] = train_df.masks.map(get_mask_type)

    # excluded all images that were below 0.2% of salt coverage
    # train_df = train_df[(train_df["coverage"] == 0)
    #                     | (train_df["coverage"] > 0.002)]
    print(f"train_df size: {len(train_df)}")

    assert phase in ("train2", "train", "finetune", "finetune2", "test")

    ids_trains = train_df.index.values
    ids_test = test_df.index.values

    if phase != "test":
        train.train_fold(train_df, ids_trains, train_dir, batch_size, n_fold,
                         phase)
        return

    all_test_preds = infer(ids_test, test_dir, batch_size // 2)

    # Submit
    # (-0.20, 0.45) (-0.28, 0.43) (-0.36, 0.41) (-0.45, 0.39)
    best_threshold = -0.28
    # best_threshold = 0
    submit_result(all_test_preds, ids_test, best_threshold)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="TGS")
    arg_parser.add_argument(
        "--n_fold",
        type=int,
        default=-1,
        help="Specific fold, -1 is all folds")
    arg_parser.add_argument(
        "--phase",
        type=str,
        default="train",
        help="Specific phase, including train, finetune, finetune2, test")
    args = arg_parser.parse_args()

    main(n_fold=args.n_fold, phase=args.phase)

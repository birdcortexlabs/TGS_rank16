#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from utils import unet_models
from utils.dataset import TGSSaltDataset, img_size_ori, unresize as un_puzzle_pad
from utils.threshold import get_best_threshold
from utils.tools import cuda, load_state, rle_encode, filter_image


def flip_tensor_lr(batch):
    columns = batch.data.size()[-1]  # 224
    return batch.index_select(3,
                              torch.LongTensor(list(reversed(
                                  range(columns)))).cuda())


def flip_tensor_ud(batch):
    rows = batch.data.size()[-2]
    return batch.index_select(2,
                              torch.LongTensor(list(reversed(
                                  range(rows)))).cuda())


def to_numpy(batch, is_sigmoid=False):
    if isinstance(batch, tuple):
        batch = batch[0]
    if is_sigmoid:
        batch = torch.sigmoid(batch)
    return batch.cpu().detach().numpy()


def evaluate(model, valid_dataloader, ids_valid):
    print("| -- Get Best Threshold from Validation Dataset -- |")

    model.eval()
    valid_preds = []
    valid_masks = []
    valid_indexs = []
    for inputs, labels, indexs in tqdm(valid_dataloader):
        inputs = cuda(inputs)
        labels = cuda(labels)

        logits = model(inputs)  # no torch.sigmoid
        logits = to_numpy(logits)

        valid_preds.append(logits)
        valid_masks.append(labels)
        valid_indexs.append(list(indexs))

    valid_preds = np.vstack(valid_preds)[:, 0, :, :]
    valid_masks = np.vstack(valid_masks)[:, 0, :, :]
    valid_preds = un_puzzle_pad(valid_preds)
    valid_masks = un_puzzle_pad(valid_masks)

    # all_valid_preds = np.vstack(all_valid_preds)
    # all_valid_masks = np.vstack(all_valid_masks)
    # all_valid_indexs = sum(all_valid_indexs, [])
    # print(all_valid_preds.shape)
    # print(all_valid_masks.shape)
    # print(set(all_valid_indexs) - set(ids_trains))

    best_threshold = get_best_threshold(valid_preds, valid_masks, ids_valid)

    return best_threshold


def predict(model, test_dataloader):
    print("| -- Predict on Test Dataset -- |")

    model.eval()
    test_preds = []
    test_indexs = []
    for inputs, indexs in tqdm(test_dataloader):
        inputs = cuda(inputs)

        logits1 = model(inputs)
        logits2 = flip_tensor_lr(model(flip_tensor_lr(inputs)))  # TTA
        logits = [logits1, logits2]
        logits = torch.mean(torch.stack(logits, 0), 0)
        logits = to_numpy(logits)

        test_preds.append(logits)
        test_indexs.append(indexs)

    test_preds = np.vstack(test_preds)[:, 0, :, :]
    test_preds = un_puzzle_pad(test_preds)

    return test_preds


def evaluate2(model, valid_dataloader, ids_valid):
    print("| -- Get Best Threshold from Validation Dataset -- |")

    model.eval()
    valid_preds = []
    valid_masks = []
    valid_indexs = []
    for inputs, labels, nemptys, indexs in tqdm(valid_dataloader):
        inputs = cuda(inputs)
        labels = cuda(labels)

        logit_fuse, logit_pixel, logit_image = model(inputs)
        logit_fuse = to_numpy(logit_fuse)

        valid_preds.append(logit_fuse)
        valid_masks.append(labels)
        valid_indexs.append(list(indexs))

    valid_preds = np.vstack(valid_preds)[:, 0, :, :]
    valid_masks = np.vstack(valid_masks)[:, 0, :, :]
    valid_preds = un_puzzle_pad(valid_preds)
    valid_masks = un_puzzle_pad(valid_masks)

    best_threshold = get_best_threshold(valid_preds, valid_masks, ids_valid)

    return best_threshold


def predict2(model, test_dataloader):
    print("| -- Predict on Test Dataset -- |")

    model.eval()
    test_preds = []
    test_indexs = []
    for inputs, indexs in tqdm(test_dataloader):
        inputs = cuda(inputs)

        logits1, _, _ = model(inputs)
        logits2, _, _ = model(flip_tensor_lr(inputs))
        logits2 = flip_tensor_lr(logits2)  # TTA
        logits = [logits1, logits2]
        logits = torch.mean(torch.stack(logits, 0), 0)
        logits = to_numpy(logits)

        test_preds.append(logits)
        test_indexs.append(indexs)

    test_preds = np.vstack(test_preds)[:, 0, :, :]
    test_preds = un_puzzle_pad(test_preds)

    return test_preds


def infer(ids_test, test_dir, batch_size):
    print(f"ids_test: {len(ids_test)}\n{ids_test[:10]}")
    test_dataset = TGSSaltDataset(test_dir, ids_test, None, mode='test')
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True)

    def get_all_test_preds(model, best_model_paths, file_path):
        if os.path.exists(file_path):
            print(f"Load {file_path}")
            all_test_preds = np.load(file_path)
            return all_test_preds

        n = len(best_model_paths)
        all_test_preds = np.zeros((n, 18000, img_size_ori, img_size_ori))

        for i, path in enumerate(best_model_paths):
            model, _, _ = load_state(model, path)
            # best_threshold = evaluate2(model, valid_dataloader, ids_valid)
            test_preds = predict2(model, test_dataloader)  # predict
            all_test_preds[i] = test_preds
        all_test_preds = all_test_preds.mean(axis=0)
        np.save(file_path, all_test_preds)
        return all_test_preds

    model = unet_models.DeepSupervision50(pretrained=True)
    model = cuda(model)

    all_test_preds = []

    # load save npys
    best_mean_npys = [
        "result/0929/finetune_meanfold.npy",
        # "result/0929/finetune2_fold0.npy",
        # "result/0929/finetune2_fold1.npy",
        # "result/0929/finetune2_fold2.npy",
        # "result/0929/finetune2_fold3.npy",
        # "result/0929/finetune2_fold4.npy",
        "result/0929/finetune2_01234.npy",
        "result/1002/finetune_meanfold.npy",
        # "result/1002/finetune2_fold0.npy",
        # "result/1002/finetune2_fold1.npy",
        # "result/1002/finetune2_fold2.npy",
        # "result/1002/finetune2_fold3.npy",
        # "result/1002/finetune2_fold4.npy",
        # "result/1002/finetune2_fold5.npy",
        # "result/1002/finetune2_fold6.npy",
        # "result/1002/finetune2_fold8.npy",
        # "result/1002/finetune2_fold9.npy",
        "result/1002/finetune2_012345689.npy",
        "result/1009/best_model_0.npy",
        "result/1009/best_model_1.npy",
        "result/1009/best_model_2.npy",
        "result/1009/best_model_3.npy",
        "result/1009/best_model_5.npy",
        "result/1009/best_model_6.npy",
        "result/1009/best_model_7.npy",
    ]
    for npy in best_mean_npys:
        test_preds = np.load(npy)
        all_test_preds.append(test_preds)

    # load and save train2 model
    # best_model_paths = []
    # for fold in [0, 1, 2, 3, 5, 6, 7]:
    #     best_model_paths.append(f"model/best_model_{fold}.pt")
    #     file_path = f"result/best_model_{fold}.npy"
    #     all_test_preds.append(
    #         get_all_test_preds(model, best_model_paths, file_path))

    # load and save finetune model
    # best_model_paths = []
    # for fold in range(4):
    #     best_model_paths.append(f"model/best_model_{fold}_finetune.pt")
    # file_path = "result/finetune_meanfold.npy"
    # all_test_preds.append(
    #     get_all_test_preds(model, best_model_paths, file_path))

    # load and save finetune2 model
    # for fold in [4, 5]:
    #     best_model_paths = []
    #     for i in range(6):
    #         best_model_paths.append(
    #             f"model/best_model_{fold}_finetune2_{i}.pt")
    #     file_path = f"result/finetune2_fold{fold}.npy"
    #     all_test_preds.append(
    #         get_all_test_preds(model, best_model_paths, file_path))

    # mean
    all_test_preds = np.array(all_test_preds)
    print(all_test_preds.shape)
    all_test_preds = all_test_preds.mean(axis=0)
    # np.save("result/1002/finetune2_012345689.npy", all_test_preds)

    return all_test_preds


def submit_result(all_test_preds, ids_test, best_threshold):
    test_bin_preds = np.int32(all_test_preds > best_threshold)

    all_masks = []
    for p_mask in tqdm(test_bin_preds):
        p_mask = rle_encode(filter_image(p_mask))
        all_masks.append(p_mask)

    all_masks2 = []
    for p_mask in tqdm(test_bin_preds):
        p_mask = rle_encode(p_mask)
        all_masks2.append(p_mask)

    submit = pd.DataFrame([ids_test, all_masks]).T
    day = time.strftime("%Y-%m-%d", time.localtime())
    submit.columns = ['id', 'rle_mask']
    result_path = f'result/submit_{day}_{best_threshold}.csv.gz'
    submit.to_csv(result_path, compression='gzip', index=False)

    submit = pd.DataFrame([ids_test, all_masks2]).T
    day = time.strftime("%Y-%m-%d", time.localtime())
    submit.columns = ['id', 'rle_mask']
    result_path = f'result/submit_{day}_{best_threshold}_no_filter.csv.gz'
    submit.to_csv(result_path, compression='gzip', index=False)

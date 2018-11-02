import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
import seaborn as sns

sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

from tqdm import tqdm_notebook
import math
from util import *
from keras.optimizers import *
import random
import keras
from keras.losses import binary_crossentropy
from scipy.misc import imsave
from sklearn.model_selection import StratifiedKFold
import sys,gc
from llose import *

img_size_ori = 101
img_size_target = 128

def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    # res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    # res[:img_size_ori, :img_size_ori] = img
    # return res


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]

def RLenc(img, order='F', format=True):
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def imagetran(x):
    x = 2 * (x - 0.5)
    return x


train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
feat_df = pd.read_csv("./f1_for_stacking.csv", index_col="id")
train_df = train_df.join(depths_df)
train_df = train_df.join(feat_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
feat_df = pd.read_csv("./f1_for_stacking_test.csv", index_col="id")
test_df = test_df.join(feat_df)
train_df["images"] = [(imagetran(np.array(
    load_img("../input/train/images/{}.png".format(idx),
             grayscale=True)) / 255.0)) for idx in
                      tqdm_notebook(train_df.index)]
train_df["images2"] = [(imagetran(np.array(
    load_img("../input/aug_masks3/{}.png".format(idx),
             grayscale=True)) / 255.0)) for idx in
                      tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(
    load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255.0
                     for idx in
                     tqdm_notebook(train_df.index)]
train_df["magic2"] = train_df.masks.map(lambda x:np.max(np.abs(np.std(x.reshape(img_size_ori,img_size_ori),axis = 0)))<=0.015)
train_df["coverage"] = train_df.masks.map(np.sum) * 1.0 / pow(img_size_ori, 2)
test_df["images"] = [(imagetran(np.array(load_img("../input/test/images/{}.png".format(idx),
                                grayscale=True)) / 255.0)) for idx in
     tqdm_notebook(test_df.index)]
test_df["images2"] = [(imagetran(np.array(load_img("../input/aug_masks3/{}.png".format(idx),
                                grayscale=True)) / 255.0)) for idx in
     tqdm_notebook(test_df.index)]
test_sample_ids = list(np.load('cache/ids_test.npy'))
index_best = (0.47, 0.001)

# index_best = (0.47 + 0.1, 0.001)
def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

def iou_metric(y_true_in, y_pred_in, print_table=False, covert=0.0):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    if np.sum(y_pred.flatten())*1.0/10201 < covert:
        y_pred[:] = 0.0

    intersection = np.sum(labels.flatten() * y_pred.flatten())
    union = np.sum(np.clip(labels.flatten() + y_pred.flatten(),0,1))
    if (np.sum(labels.flatten()) == 0):
        if (np.sum(y_pred.flatten()) == 0):
            iou = 1
        else:
            iou = 0
    else:
        iou = intersection * 1.0 / union
    iou = [[iou]]
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        # print (t,tp,fp,fn)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in,covert = 0.0):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch], covert = covert)
        metric.append(value)
    return np.mean(metric)

#
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
print(train_df["coverage_class"].value_counts()/4000)
indexs = []
images = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
masks = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
labels = train_df.coverage_class

# unet8
model_count = 0
train_df["unet8l"] = train_df["images"].copy()
train_df["unet8m"] = train_df["images"].copy()
test_df["unet8l"] = test_df["images"].copy()
test_df["unet8m"] = test_df["images"].copy()
predsm = []
predsl = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
for train_idx, val_idx in skf.split(images, labels):
    print(model_count)
    model_count += 1
    ids_train, ids_valid= \
        train_df.index.values[train_idx], train_df.index.values[val_idx]

    unet8l_np = np.fromfile("cache/unet8lov_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet8mse_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    train_df.loc[ids_valid,"unet8l"] = [unet8l_np[x] for x in range(len(ids_valid))]
    train_df.loc[ids_valid, "unet8m"] = [unet8m_np[x] for x in range(len(ids_valid))]
    unet8l_np = np.fromfile("cache/unet8lov_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet8mse_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    predsl.append(unet8l_np)
    predsm.append(unet8m_np)

predsl = sum(predsl)/len(predsl)
predsm = sum(predsm)/len(predsm)
test_df.loc[:,"unet8l"] = [predsl[x] for x in range(test_df.index.shape[0])]
test_df.loc[:,"unet8m"] = [predsm[x] for x in range(test_df.index.shape[0])]
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet8l"].values.tolist()) > index_best[0]), index_best[1])
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet8m"].values.tolist()) > index_best[0]), index_best[1])
# print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet8m"].values.tolist()) + np.array(train_df["unet8l"].values.tolist()) > index_best[0] * 2), index_best[1])

# guido1
def sigmoid(x):
    return 1/(1+math.e**-x)

model_count = 0
train_df["guido1"] = train_df["images"].copy()
test_df["guido1"] = test_df["images"].copy()
predsl = []
for model_count in range(8):
    ids_valid = list(np.load('cache/ids_valid_{}.npy'.format(model_count)))
    unet8l_np = sigmoid(np.load('cache/valid_preds_{}.npy'.format(model_count))).reshape(-1, img_size_ori, img_size_ori, 1)
    print(model_count)
    train_df.loc[ids_valid,"guido1"] = [unet8l_np[x] for x in range(len(ids_valid))]
    unet8l_np = sigmoid(np.load('cache/best_model_{}.npy'.format(model_count))).reshape(-1, img_size_ori, img_size_ori, 1)
    predsl.append(unet8l_np)

predsl = sum(predsl)/len(predsl)
test_df.loc[test_sample_ids,"guido1"] = [predsl[x] for x in range(len(test_sample_ids))]
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["guido1"].values.tolist()) > index_best[0]), index_best[1])

# inception
model_count = 0
train_df["inception2l"] = train_df["images"].copy()
train_df["inception2m"] = train_df["images"].copy()
test_df["inception2l"] = test_df["images"].copy()
test_df["inception2m"] = test_df["images"].copy()
predsm = []
predsl = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
for train_idx, val_idx in skf.split(images, labels):
    print(model_count)
    model_count += 1
    ids_train, ids_valid= \
        train_df.index.values[train_idx], train_df.index.values[val_idx]

    unet8l_np = np.fromfile("cache/inception2lov_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/inception2mse_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    train_df.loc[ids_valid,"inception2l"] = [unet8l_np[x] for x in range(len(ids_valid))]
    train_df.loc[ids_valid, "inception2m"] = [unet8m_np[x] for x in range(len(ids_valid))]
    unet8l_np = np.fromfile("cache/inception2lov_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/inception2mse_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    predsl.append(unet8l_np)
    predsm.append(unet8m_np)

predsl = sum(predsl)/len(predsl)
predsm = sum(predsm)/len(predsm)
test_df.loc[:,"inception2l"] = [predsl[x] for x in range(test_df.index.shape[0])]
test_df.loc[:,"inception2m"] = [predsm[x] for x in range(test_df.index.shape[0])]
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["inception2l"].values.tolist()) > index_best[0]), index_best[1])
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["inception2m"].values.tolist()) > index_best[0]), index_best[1])


# unet10
model_count = 0
train_df["unet10l"] = train_df["images"].copy()
train_df["unet10m"] = train_df["images"].copy()
test_df["unet10l"] = test_df["images"].copy()
test_df["unet10m"] = test_df["images"].copy()
predsm = []
predsl = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(images, labels):
    print(model_count)
    model_count += 1
    ids_train, ids_valid= \
        train_df.index.values[train_idx], train_df.index.values[val_idx]

    unet8l_np = np.fromfile("cache/unet10lov_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet10mse_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    train_df.loc[ids_valid,"unet10l"] = [unet8l_np[x] for x in range(len(ids_valid))]
    train_df.loc[ids_valid, "unet10m"] = [unet8m_np[x] for x in range(len(ids_valid))]
    unet8l_np = np.fromfile("cache/unet10lov_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet10mse_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    predsl.append(unet8l_np)
    predsm.append(unet8m_np)

predsl = sum(predsl)/len(predsl)
predsm = sum(predsm)/len(predsm)
test_df.loc[:,"unet10l"] = [predsl[x] for x in range(test_df.index.shape[0])]
test_df.loc[:,"unet10m"] = [predsm[x] for x in range(test_df.index.shape[0])]
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet10l"].values.tolist()) > index_best[0]), index_best[1])
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet10m"].values.tolist()) > index_best[0]), index_best[1])

# unet15
model_count = 0
train_df["unet15l"] = train_df["images"].copy()
train_df["unet15m"] = train_df["images"].copy()
test_df["unet15l"] = test_df["images"].copy()
test_df["unet15m"] = test_df["images"].copy()
predsm = []
predsl = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(images, labels):
    print(model_count)
    model_count += 1
    ids_train, ids_valid= \
        train_df.index.values[train_idx], train_df.index.values[val_idx]

    unet8l_np = np.fromfile("cache/unet15lov_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet15mse_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    train_df.loc[ids_valid,"unet15l"] = [unet8l_np[x] for x in range(len(ids_valid))]
    train_df.loc[ids_valid, "unet15m"] = [unet8m_np[x] for x in range(len(ids_valid))]
    unet8l_np = np.fromfile("cache/unet15lov_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet15mse_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    predsl.append(unet8l_np)
    predsm.append(unet8m_np)

predsl = sum(predsl)/len(predsl)
predsm = sum(predsm)/len(predsm)
test_df.loc[:,"unet15l"] = [predsl[x] for x in range(test_df.index.shape[0])]
test_df.loc[:,"unet15m"] = [predsm[x] for x in range(test_df.index.shape[0])]
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet15l"].values.tolist()) > index_best[0]), index_best[1])
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet15m"].values.tolist()) > index_best[0]), index_best[1])


# unet5
model_count = 0
train_df["unet5l"] = train_df["images"].copy()
train_df["unet5m"] = train_df["images"].copy()
test_df["unet5l"] = test_df["images"].copy()
test_df["unet5m"] = test_df["images"].copy()
predsm = []
predsl = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
for train_idx, val_idx in skf.split(images, labels):
    print(model_count)
    model_count += 1
    ids_train, ids_valid= \
        train_df.index.values[train_idx], train_df.index.values[val_idx]

    unet8l_np = np.fromfile("cache/unet5lov_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet5mse_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    train_df.loc[ids_valid,"unet5l"] = [unet8l_np[x] for x in range(len(ids_valid))]
    train_df.loc[ids_valid, "unet5m"] = [unet8m_np[x] for x in range(len(ids_valid))]
    unet8l_np = np.fromfile("cache/unet5lov_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet5mse_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    predsl.append(unet8l_np)
    predsm.append(unet8m_np)

predsl = sum(predsl)/len(predsl)
predsm = sum(predsm)/len(predsm)
test_df.loc[:,"unet5l"] = [predsl[x] for x in range(test_df.index.shape[0])]
test_df.loc[:,"unet5m"] = [predsm[x] for x in range(test_df.index.shape[0])]
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet5l"].values.tolist()) > index_best[0]), index_best[1])
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet5m"].values.tolist()) > index_best[0]), index_best[1])

# unet11
model_count = 0
train_df["unet11l"] = train_df["images"].copy()
train_df["unet11m"] = train_df["images"].copy()
test_df["unet11l"] = test_df["images"].copy()
test_df["unet11m"] = test_df["images"].copy()
predsm = []
predsl = []
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
train_df["coverage"] = train_df.masks.map(np.sum) / pow(192, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
labels = train_df.coverage_class
for train_idx, val_idx in skf.split(images, labels):
    print(model_count)
    model_count += 1
    ids_train, ids_valid= \
        train_df.index.values[train_idx], train_df.index.values[val_idx]

    unet8l_np = np.fromfile("cache/unet11lov_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet11mse_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    train_df.loc[ids_valid,"unet11l"] = [unet8l_np[x] for x in range(len(ids_valid))]
    train_df.loc[ids_valid, "unet11m"] = [unet8m_np[x] for x in range(len(ids_valid))]
    unet8l_np = np.fromfile("cache/unet11lov_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    unet8m_np = np.fromfile("cache/unet11mse_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
    predsl.append(unet8l_np)
    predsm.append(unet8m_np)

predsl = sum(predsl)/len(predsl)
predsm = sum(predsm)/len(predsm)
test_df.loc[:,"unet11l"] = [predsl[x] for x in range(test_df.index.shape[0])]
test_df.loc[:,"unet11m"] = [predsm[x] for x in range(test_df.index.shape[0])]
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet11l"].values.tolist()) > index_best[0]), index_best[1])
print "merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet11m"].values.tolist()) > index_best[0]), index_best[1])

# model_count = 0
# train_df["unet13l"] = train_df["images"].copy()
# train_df["unet13m"] = train_df["images"].copy()
# test_df["unet13l"] = test_df["images"].copy()
# test_df["unet13m"] = test_df["images"].copy()
# predsm = []
# predsl = []
# for train_idx, val_idx in skf.split(images, labels):
#     print(model_count)
#     model_count += 1
#     ids_train, ids_valid= \
#         train_df.index.values[train_idx], train_df.index.values[val_idx]
#
#     unet8l_np = np.fromfile("cache/unet13lov_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
#     unet8m_np = np.fromfile("cache/unet13mse_"+str(model_count)+"_valid_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
#     train_df.loc[ids_valid,"unet13l"] = [unet8l_np[x] for x in range(len(ids_valid))]
#     train_df.loc[ids_valid, "unet13m"] = [unet8m_np[x] for x in range(len(ids_valid))]
#     unet8l_np = np.fromfile("cache/unet13lov_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
#     unet8m_np = np.fromfile("cache/unet13mse_"+str(model_count)+"_np",dtype = np.float32).reshape(-1, img_size_ori, img_size_ori, 1)
#     predsl.append(unet8l_np)
#     predsm.append(unet8m_np)
#
# predsl = sum(predsl)/len(predsl)
# predsm = sum(predsm)/len(predsm)
# test_df.loc[:,"unet13l"] = [predsl[x] for x in range(test_df.index.shape[0])]
# test_df.loc[:,"unet13m"] = [predsm[x] for x in range(test_df.index.shape[0])]
# print "13merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet13l"].values.tolist()) > index_best[0]), index_best[1])
# print "13merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet13m"].values.tolist()) > index_best[0]), index_best[1])
# print "13merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet13l"].values.tolist()) > index_best[0] + 0.05), index_best[1])
# print "13merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet13m"].values.tolist()) > index_best[0] + 0.05), index_best[1])
# print "13merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet13l"].values.tolist()) > index_best[0] + 0.15), index_best[1])
# print "13merge_np", iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(np.array(train_df["unet13m"].values.tolist()) > index_best[0] + 0.15), index_best[1])


base_np = np.array(train_df["unet8l"].values.tolist()) * 0.5 + np.array(
            train_df["unet10l"].values.tolist()) + np.array(
            train_df["unet8m"].values.tolist()) * 0.5 + np.array(
            train_df["unet10m"].values.tolist()) + np.array(
            train_df["unet5m"].values.tolist()) + np.array(
            train_df["unet5l"].values.tolist()) + np.array(
            train_df["guido1"].values.tolist()) * 2
print 'base:', iou_metric_batch(
        np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
        np.int32(base_np > (index_best[0] - 0.05) * 7.0), index_best[1])
print 'base:', iou_metric_batch(
        np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
        np.int32(base_np > (index_best[0] - 0.15) * 7.0), index_best[1])
print 'base:', iou_metric_batch(
        np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
        np.int32(base_np > (index_best[0] + 0.05) * 7.0), index_best[1])
print 'base:',iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(base_np > index_best[0] * 7.0), index_best[1])


base_np = np.array(train_df["unet8l"].values.tolist()) * 0.5 + np.array(
            train_df["unet10l"].values.tolist()) + np.array(
            train_df["unet8m"].values.tolist()) * 0.5 + np.array(
            train_df["unet10m"].values.tolist()) + np.array(
            train_df["unet5m"].values.tolist()) + np.array(
            train_df["unet5l"].values.tolist()) + np.array(
            train_df["guido1"].values.tolist()) * 2 + np.array(
            train_df["unet15m"].values.tolist()) * 0.5 + np.array(
            train_df["unet15l"].values.tolist()) * 0.5 + np.array(
            train_df["unet11m"].values.tolist())  + np.array(
            train_df["unet11l"].values.tolist())
print 'base:', iou_metric_batch(
        np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
        np.int32(base_np > (index_best[0] - 0.05) * 10.0), index_best[1])
print 'base:', iou_metric_batch(
        np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
        np.int32(base_np > (index_best[0] - 0.15) * 10.0), index_best[1])
print 'base:', iou_metric_batch(
        np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
        np.int32(base_np > (index_best[0] + 0.05) * 10.0), index_best[1])
print 'base:',iou_metric_batch(np.int32(np.array(train_df["masks"].values.tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(base_np > index_best[0] * 10.0), index_best[1])

exit()

class LossHistory(keras.callbacks.Callback):
    def __init__(self, X_val, y_val, model,thre= 0.5):
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.thre = thre

    def on_epoch_end(self, epoch, logs={}):
        thres = self.thre
        preds_valid = self.model.predict(self.X_val)
        preds_valid = np.array([downsample(x) for x in preds_valid])
        y_valid = np.array(self.y_val)
        iou1 = iou_metric_batch(np.int32(y_valid > 0.45), np.int32(preds_valid> thres - 0.05),0.001)
        logs['iou'] = iou1
        print('val iou:{},mean:{}'.format(iou1,np.mean(preds_valid)))
        iou2 = iou_metric_batch(np.int32(y_valid > 0.55), np.int32(preds_valid> thres + 0.05),0.001)
        logs['iou'] = iou2
        print('val iou:{},mean:{}'.format(iou2,np.mean(preds_valid)))
        iou = iou_metric_batch(np.int32(y_valid > 0.5), np.int32(preds_valid> thres),0.001)
        logs['iou'] = (iou + 0.2 * iou1 + 0.2 * iou2)/1.4
        print('val iou:{},mean:{}'.format(iou,np.mean(preds_valid)))
        del preds_valid
        gc.collect()

def mycustom(y_true, y_pred):
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    y_union = K.clip(y_true+y_pred,1e-5,1)
    y_weight = K.clip(img_size_target * img_size_target * 1.0 / K.sum(y_union, axis=-1), 1.0, 80.0)
    return K.mean(K.square(K.abs(y_pred - y_true)+0.15), axis=-1) * y_weight

def mycustom2(y_true, y_pred):
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    return K.mean(lovasz_hinge(y_pred,y_true))

def sigmoid(x):
    return 1/(1+math.e**-x)

def fsigmoid(x):
    retuen -math.log(1/x-1)


images = np.concatenate([np.array(train_df.unet8l.tolist())+np.array(train_df.unet8m.tolist()),
                         np.array(train_df.inception2l.tolist())+np.array(train_df.inception2m.tolist()),
                         np.array(train_df.unet10l.tolist())+np.array(train_df.unet10m.tolist()),
                         np.array(train_df.unet5l.tolist())+np.array(train_df.unet5m.tolist()),
                         np.array(train_df.unet11l.tolist()) + np.array(train_df.unet11m.tolist()),np.array(train_df.guido1.tolist())
                            , np.array(train_df.images2.tolist()).reshape(-1, img_size_ori, img_size_ori, 1)
                         ],axis = -1).reshape(-1, img_size_ori, img_size_ori, 7)
x_test_image = np.concatenate([np.array(test_df.unet8l.tolist())+np.array(test_df.unet8m.tolist()),
                         np.array(test_df.inception2l.tolist())+np.array(test_df.inception2m.tolist()),
                         np.array(test_df.unet10l.tolist())+np.array(test_df.unet10m.tolist()),
                         np.array(test_df.unet5l.tolist())+np.array(test_df.unet5m.tolist()),
                               np.array(test_df.unet11l.tolist()) + np.array(test_df.unet11m.tolist()),np.array(test_df.guido1.tolist()),
                         np.array(test_df.images2.tolist()).reshape(-1, img_size_ori, img_size_ori, 1)
                               ],axis = -1).reshape(-1, img_size_ori, img_size_ori, 7)
masks = np.array(train_df.masks.tolist()).reshape(-1, img_size_ori, img_size_ori, 1)
model_count = 0
preds = np.zeros((18000,101,101,1))
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
labels = train_df.coverage_class
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
# for train_idx, val_idx in skf.split(images, labels):
#     ids_train, ids_valid, x_train, x_valid, y_train, y_valid = \
#         train_df.index.values[train_idx], train_df.index.values[val_idx], \
#         images[train_idx], images[val_idx], \
#         masks[train_idx], masks[val_idx], \
#
#     print model_count
#
#     input_layer = Input((img_size_ori, img_size_ori, 7), dtype='float32')
#     uconv1 = BatchNormalization()(input_layer)
#     uconv1 = concatenate([Conv2D(64, (3, 3), padding="same", activation="relu")(uconv1),
#                           Conv2D(8, (3, 3), padding="same", activation="sigmoid")(uconv1)])
#     # uconv1 = Conv2D(64, (3, 3), padding="same", activation="relu")(input_layer)
#     uconv1 = BatchNormalization()(uconv1)
#     uconv1 = Dropout(0.2)(uconv1)
#     uconv1 = concatenate([Conv2D(16, (3, 3), padding="same", activation="relu")(uconv1),
#                           Conv2D(8, (3, 3), padding="same", activation="sigmoid")(uconv1)])
#     uconv1 = BatchNormalization()(uconv1)
#     # uconv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(uconv1)
#     # uconv1 = Conv2D(16, (3, 3), padding="same", activation="relu")(uconv1)
#     uconv1 = Dropout(0.2)(uconv1)
#     output_layer = Conv2D(1, (1, 1), padding="same",
#                           name='main_output', activation="sigmoid", use_bias = False)(uconv1)
#     model = Model([input_layer], output_layer)
#     modelname = 'cache/stacking' + str(model_count) + '.h5'
#     model.compile(loss=mycustom, optimizer='adam',
#                       metrics=["accuracy"])
#     lh = LossHistory([x_valid], y_valid, model)
#     early_stopping = EarlyStopping(patience=10, verbose=1, mode='max',
#                                        monitor='iou')
#     model_checkpoint = ModelCheckpoint(modelname, save_weights_only=True, save_best_only=True, verbose=1, mode='max',
#                                        monitor='iou')
#     reduce_lr = ReduceLROnPlateau(factor=0.4, patience=4, min_lr=0.00005, verbose=1, mode='max',
#                                        monitor='iou')
#     base_np = np.array(train_df["unet8l"].values[val_idx].tolist()) * 0.5 + np.array(
#             train_df["unet10l"].values[val_idx].tolist()) + np.array(
#             train_df["unet8m"].values[val_idx].tolist()) * 0.5 + np.array(
#             train_df["unet10m"].values[val_idx].tolist()) + np.array(
#             train_df["unet5m"].values[val_idx].tolist()) + np.array(
#             train_df["unet5l"].values[val_idx].tolist()) + np.array(
#             train_df["guido1"].values[val_idx].tolist()) * 2
#     print 'base:', iou_metric_batch(
#         np.int32(np.array(train_df["masks"].values[val_idx].tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
#         np.int32(base_np > (index_best[0] - 0.05) * 7.0), index_best[1])
#     print 'base:', iou_metric_batch(
#         np.int32(np.array(train_df["masks"].values[val_idx].tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
#         np.int32(base_np > (index_best[0] - 0.15) * 7.0), index_best[1])
#     print 'base:', iou_metric_batch(
#         np.int32(np.array(train_df["masks"].values[val_idx].tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
#         np.int32(base_np > (index_best[0] + 0.05) * 7.0), index_best[1])
#     print 'base:',iou_metric_batch(np.int32(np.array(train_df["masks"].values[val_idx].tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(base_np > index_best[0] * 7.0), index_best[1])
#     model.fit([x_train], y_train,
#               validation_data=[[x_valid], y_valid],
#               epochs=100,
#               batch_size=64,
#               callbacks=[lh,model_checkpoint,early_stopping,reduce_lr])
#     model_count += 1
#     #
#     model.load_weights(modelname)
#     lh.on_epoch_end(0)
#     preds2 = model.predict(x_test_image,batch_size = 100)
#     preds2 = np.array([downsample(x) for x in preds2], dtype=np.float32)
#     preds += preds2

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
for train_idx, val_idx in skf.split(images, labels):
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid = \
        train_df.index.values[train_idx], train_df.index.values[val_idx], \
        images[train_idx], images[val_idx], \
        masks[train_idx], masks[val_idx], \

    print model_count

    input_layer = Input((img_size_ori, img_size_ori, 7), dtype='float32')
    uconv1 = BatchNormalization()(input_layer)
    # uconv1 = Conv2D(64, (3, 3), padding="same", activation="relu")(input_layer)
    uconv1 = concatenate([Conv2D(64, (3, 3), padding="same", activation="relu")(uconv1),
                              Conv2D(8, (3, 3), padding="same", activation="sigmoid")(uconv1)])
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Dropout(0.2)(uconv1)
    uconv1 = concatenate([Conv2D(16, (3, 3), padding="same", activation="relu")(uconv1),
                          Conv2D(8, (3, 3), padding="same", activation="sigmoid")(uconv1)])
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Dropout(0.2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same",
                          name='main_output', activation="sigmoid", use_bias = False)(uconv1)
    model = Model([input_layer], output_layer)
    modelname = 'cache/stacking' + str(model_count) + '.h5'
    model.compile(loss=mycustom, optimizer='adam',
                      metrics=["accuracy"])
    lh = LossHistory([x_valid], y_valid, model)
    early_stopping = EarlyStopping(patience=10, verbose=1, mode='max',
                                       monitor='iou')
    model_checkpoint = ModelCheckpoint(modelname, save_weights_only=True, save_best_only=True, verbose=1, mode='max',
                                       monitor='iou')
    reduce_lr = ReduceLROnPlateau(factor=0.4, patience=4, min_lr=0.00005, verbose=1, mode='max',
                                       monitor='iou')
    base_np = np.array(train_df["unet8l"].values[val_idx].tolist()) * 0.5 + np.array(
            train_df["unet10l"].values[val_idx].tolist()) + np.array(
            train_df["unet8m"].values[val_idx].tolist()) * 0.5 + np.array(
            train_df["unet10m"].values[val_idx].tolist()) + np.array(
            train_df["unet5m"].values[val_idx].tolist()) + np.array(
            train_df["unet5l"].values[val_idx].tolist()) + np.array(
            train_df["guido1"].values[val_idx].tolist()) * 2
    print 'base:', iou_metric_batch(
        np.int32(np.array(train_df["masks"].values[val_idx].tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
        np.int32(base_np > (index_best[0] - 0.05) * 7.0), index_best[1])
    print 'base:', iou_metric_batch(
        np.int32(np.array(train_df["masks"].values[val_idx].tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
        np.int32(base_np > (index_best[0] - 0.15) * 7.0), index_best[1])
    print 'base:', iou_metric_batch(
        np.int32(np.array(train_df["masks"].values[val_idx].tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1),
        np.int32(base_np > (index_best[0] + 0.05) * 7.0), index_best[1])
    print 'base:',iou_metric_batch(np.int32(np.array(train_df["masks"].values[val_idx].tolist()) > 0.5).reshape(-1, img_size_ori, img_size_ori, 1), np.int32(base_np > index_best[0] * 7.0), index_best[1])
    model.fit([x_train], y_train,
              validation_data=[[x_valid], y_valid],
              epochs=100,
              batch_size=64,
              callbacks=[lh,model_checkpoint,early_stopping,reduce_lr])
    model_count += 1
    #
    model.load_weights(modelname)
    lh.on_epoch_end(0)
    preds2 = model.predict(x_test_image,batch_size = 100)
    preds2 = np.array([downsample(x) for x in preds2], dtype=np.float32)
    preds += preds2

preds2 = preds/13
valid_fix_up = {}
valid_fix_down = {}
valid_fix2 = {}
count = 0
for line in open('../feat/debugpingtu1'):
    if not line:
        continue
    array = line.strip().split(' ')
    id1 = array[1]
    id2 = array[0]
    score = float(array[5])
    if score > 0.3:
        if id1 not in test_df.index and train_df['magic2'][id1]:
            if np.sum(train_df['masks'][id1]) == 0:
                valid_fix_up[id2] = train_df['masks'][id1].reshape(img_size_ori, img_size_ori, 1).astype(np.int32)
            else:
                valid_fix2[id2] = train_df['masks'][id1].reshape(img_size_ori, img_size_ori, 1).astype(np.int32)

        if id2 not in test_df.index and train_df['magic2'][id2]:
            if np.sum(train_df['masks'][id2]) == 0:
                valid_fix_down[id1] = train_df['masks'][id2].reshape(img_size_ori, img_size_ori, 1).astype(np.int32)
            else:
                valid_fix2[id1] = train_df['masks'][id2].reshape(img_size_ori, img_size_ori, 1).astype(np.int32)

print(len(valid_fix_up.items()))
print(len(valid_fix_down.items()))
print(len(valid_fix2.items()))

valudcounts = {}
for i in range(test_df.index.values.shape[0]):
    if (test_df.index.values[i]) in valid_fix2.keys():
        preds2[i] = valid_fix2[test_df.index.values[i]] + preds2[i] * 0.6
    if (test_df.index.values[i]) in valid_fix_up.keys() or (test_df.index.values[i]) in valid_fix_down.keys():
        preds2[i] *= 0.8
    preds2[i] = np.int32(preds2[i] > index_best[0])
    if np.sum(preds2[i].flatten()) * 1.0 / 10201 < index_best[1]:
        preds2[i][:] = 0.0
    valudcounts[math.ceil(np.sum(preds2[i].flatten()) * 10.0 / 10201)] = valudcounts.get(math.ceil(np.sum(preds2[i].flatten()) * 10.0 / 10201),0) + 1.0


print(map(lambda x:(x[0],x[1]*1.0/18000),valudcounts.items()))
preds2 = np.array(preds2)
preds2.astype(np.float32).tofile("cache/submit_stacking_np")
pred_dict = {idx: RLenc((preds2[i])) for i, idx in
                 enumerate(tqdm_notebook(test_df.index.values))}
sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv("unet_incep_leak_stacking_sub8.csv")
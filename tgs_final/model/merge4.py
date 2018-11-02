import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
import seaborn as sns

sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img,save_img
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
import sys,gc,json
import cPickle as pkl

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

model_count = 0
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
feat_df = pd.read_csv("./f1_for_stacking.csv", index_col="id")
feat_df = feat_df.join(pd.read_csv("./f2_for_stacking.csv", index_col="id"))
train_df = train_df.join(depths_df)
train_df = train_df.join(feat_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
feat_df = pd.read_csv("./f1_for_stacking_test.csv", index_col="id")
feat_df = feat_df.join(pd.read_csv("./f2_for_stacking_test.csv", index_col="id"))
test_df = test_df.join(feat_df)
train_df["masks"] = [np.array(
    load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255.0
                     for idx in
                     tqdm_notebook(train_df.index)]
sub_np = np.fromfile("../model/cache/submit_np",dtype = np.float32).reshape(-1,101,101,1)
test_df["masks"] = [(sub_np[ii].reshape(img_size_ori,img_size_ori))  for ii in
                      range(test_df.index.shape[0])]
train_df["magic2"] = train_df.masks.map(lambda x:np.max(np.abs(np.std(x.reshape(img_size_ori,img_size_ori),axis = 0)))<=0.015)
train_df["magic1"] = train_df.masks.map(lambda x:np.max(np.abs(np.std(x.reshape(img_size_ori,img_size_ori)[50:,:],axis = 0)))<=0.05)
train_df["magic2"] = train_df.masks.map(lambda x:np.max(np.abs(np.std(x.reshape(img_size_ori,img_size_ori)[:50,:],axis = 0)))<=0.05)
# for i in range(train_df.shape[0]):
#     if train_df["magic2"].values[i] and np.sum(np.int32(train_df['masks'].values[i] > 0.5)) != 0:
#         print np.max(np.abs(np.std(train_df['masks'].values[i].reshape(img_size_ori,img_size_ori),axis = 0))),\
#             np.mean(np.abs(np.std(train_df['masks'].values[i].reshape(img_size_ori,img_size_ori),axis = 0))),\
#             np.where(np.abs(np.std(train_df['masks'].values[i].reshape(img_size_ori,img_size_ori),axis = 0))!=0)[0].shape
#
# train_df["magic2"] = train_df.masks.map(lambda x:np.max(np.abs(np.std(x.reshape(img_size_ori,img_size_ori),axis = 0)))<=0.05 and
#                                                  np.mean(np.abs(np.std(x.reshape(img_size_ori,img_size_ori),axis = 0)))<=0.005)

train_df["coverage"] = train_df.masks.map(np.sum) * 1.0 / pow(img_size_ori, 2)
train_union_df = train_df.append(test_df)
index_best = (0.47, 0.000)
# index_best = (0.47 + 0.1, 0.001)
def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

#
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
print(train_df["coverage_class"].value_counts()/4000)
indexs = []

# print(train_df["magic2"]['d2522cfc93'])
# exit()

def sigmoid(x):
    return 1/(1+math.e**-x)

def fsigmoid(x):
    retuen -math.log(1/x-1)

valid_fix_up = {} # vec up
valid_fix2_up = {} # vec up infer
valid_fix3_up = {} # vec up infer test
valid_fix_down = {} # vec down
valid_fix2_down = {} # all zero down
valid_fix3_down = {} # vec down infer
valid_fix4_down = {} # all one down
valid_fix6_down = {} # all zero down
valid_fix_left = {} # left
valid_fix_right = {} # right
count = 0

with open('../feat/puzzle.pkl', 'r') as fr:
    results = pkl.load(fr)
results = [json.loads(_) for _ in results]

for graph_s,graph_id in results:
    graph_id = np.array(graph_id).reshape(graph_s)
    for j in range(1, graph_s[1] - 1):
        up_bound = np.zeros(img_size_ori)
        up_i = 0
        down_bound = np.zeros(img_size_ori)
        isbreak = False
        for i in range(1, graph_s[0] - 1):
            if graph_id[i, j] != "0000000000" and np.sum(np.int32(train_union_df['masks'][graph_id[i, j]][-1,:] > 0.65)) > 95:
                isbreak = True
                break
        if isbreak:
            continue
        for i in range(1, graph_s[0] - 1):
            if graph_id[i, j] != "0000000000" and np.sum(np.abs(np.int32(train_union_df['masks'][graph_id[i, j]][50,:] > 0.65) - np.int32(train_union_df['masks'][graph_id[i, j]][-1,:] > 0.65))) < 10 and 10 < np.sum(np.int32(train_union_df['masks'][graph_id[i, j]][-1,:] > 0.5)) < 90:
                if np.sum(up_bound) == 0:
                    up_bound = np.int32(train_union_df['masks'][graph_id[i, j]][-1,:] > 0.65)
                    up_i = i
                elif np.sum(np.abs(np.int32(train_union_df['masks'][graph_id[i, j]][-1,:] > 0.65) - up_bound)) < 10:
                    down_bound = np.int32(train_union_df['masks'][graph_id[i, j]][-1,:] > 0.65)
                    for ii in range(up_i + 1,i):
                        if ii <= graph_s[0] - 1 and graph_id[ii, j] != "0000000000":
                            valid_fix2_up[graph_id[ii, j]] = np.repeat(up_bound.reshape(1,img_size_ori),img_size_ori,axis = 0).reshape(img_size_ori, img_size_ori,
                                                                                                                                       1).astype(np.int32)
                    for ii in range(i+1, i+6):
                        if ii <= graph_s[0] - 1 and graph_id[ii, j] != "0000000000":
                            valid_fix2_up[graph_id[ii, j]] = np.repeat(down_bound.reshape(1, img_size_ori),
                                                                       img_size_ori, axis=0).reshape(img_size_ori,
                                                                                                     img_size_ori,
                                                                                                     1).astype(
                                np.int32)
                elif np.sum((np.int32(train_union_df['masks'][graph_id[i, j]][-1,:] > 0.65) - up_bound)) > 0:
                    for ii in range(i + 1, i + 6):
                        if ii <= graph_s[0] - 1 and graph_id[ii, j] != "0000000000":
                            valid_fix2_up[graph_id[ii, j]] = np.repeat(
                                up_bound.reshape(1, img_size_ori),
                                img_size_ori, axis=0).reshape(img_size_ori,
                                                              img_size_ori,
                                                              1).astype(
                                np.int32)


    for i in range(1,graph_s[0]-1):
        for j in range(1, graph_s[1] - 1):
            if graph_id[i, j] != "0000000000":
                if np.sum(np.int32(train_union_df['masks'][graph_id[i, j]][-2:,:] > 0.5)) == 101 * 2:
                    for ii in range(i + 1, i + 6):
                        if ii <= graph_s[0] - 1 and graph_id[ii, j] != "0000000000":
                            valid_fix4_down[graph_id[ii, j]] = ii - i

                if np.sum(np.int32(train_union_df['masks'][graph_id[i, j]] > 0.5)) == 0:
                    for iii in range(0, i):
                        if graph_id[iii, j] != "0000000000" and np.sum(np.int32(train_union_df['masks'][graph_id[iii, j]][-1,:] > 0.5)) >= 90:
                            for ii in range(i + 1, i + 6):
                                if ii <= graph_s[0] - 1 and graph_id[ii, j] != "0000000000":
                                    valid_fix2_down[graph_id[ii, j]] = ii - i
                            break
                    if graph_s[0] >= 8 and graph_s[0] - i <= 5 and graph_id[i, j] not in test_df.index:
                        for ii in range(i + 1, i + 6):
                            if ii <= graph_s[0] - 1 and graph_id[ii, j] != "0000000000":
                                valid_fix6_down[graph_id[ii, j]] = ii

            if graph_id[i, j] in test_df.index or graph_id[i, j] == "0000000000":
                continue

            if graph_id[i, j - 1] != "0000000000" and np.sum(train_df['masks'][graph_id[i, j]][:,0]) == 101:
                # print(graph_id[i, j])
                valid_fix_left[graph_id[i, j - 1]] = train_df['masks'][graph_id[i, j]].reshape(img_size_ori, img_size_ori, 1).astype(np.int32)
            if graph_id[i, j + 1] != "0000000000" and np.sum(train_df['masks'][graph_id[i, j]][:,-1]) == 101:
                valid_fix_right[graph_id[i, j + 1]] = train_df['masks'][graph_id[i, j]].reshape(img_size_ori, img_size_ori,
                                                                                             1).astype(np.int32)
            if graph_id[i - 1, j] != "0000000000" and train_df['magic2'][graph_id[i, j]] and np.sum(train_df['masks'][graph_id[i, j]][:50,:]) != 0 and np.sum(train_df['masks'][graph_id[i, j]][:50,:]) != 50 * 101:
                # valid_fix_up[graph_id[i - 1, j]] = train_df['masks'][graph_id[i, j]].reshape(img_size_ori, img_size_ori, 1).astype(np.int32)
                valid_fix_up[graph_id[i - 1, j]] = np.repeat(train_df['masks'][graph_id[i, j]][0,:].reshape(1,img_size_ori),img_size_ori,axis = 0).reshape(img_size_ori, img_size_ori,
                                                                                             1).astype(np.int32)
            if graph_id[i + 1, j] != "0000000000" and train_df['magic1'][graph_id[i, j]] and np.sum(train_df['masks'][graph_id[i, j]][-50:,:]) != 0 and np.sum(train_df['masks'][graph_id[i, j]][-50:,:]) != 50 * 101:
                valid_fix_down[graph_id[i + 1, j]] = np.repeat(
                    train_df['masks'][graph_id[i, j]][-1, :].reshape(1, img_size_ori), img_size_ori,
                    axis=0).reshape(img_size_ori, img_size_ori,
                                    1).astype(np.int32)
            if train_df['magic1'][graph_id[i, j]] and np.sum(train_df['masks'][graph_id[i, j]][-50:,:]) != 0 and np.sum(train_df['masks'][graph_id[i, j]][-50:,:]) != 50 * 101:
                for ii in range(i + 1, i + 6):
                    if ii <= graph_s[0]-1 and graph_id[ii, j] != "0000000000":
                        valid_fix3_down[graph_id[ii, j]] =  np.repeat(
                    train_df['masks'][graph_id[i, j]][-1, :].reshape(1, img_size_ori), img_size_ori,
                    axis=0).reshape(img_size_ori, img_size_ori,
                                    1).astype(np.int32)






print(len(set(valid_fix_up.keys()) & set(test_df.index.values)))
print(len(set(valid_fix2_up.keys()) & set(test_df.index.values)))
print(len(set(valid_fix_down.keys()) & set(test_df.index.values)))
print(len(set(valid_fix_left.keys()) & set(test_df.index.values)))
print(len(set(valid_fix_right.keys()) & set(test_df.index.values)))
print(len(set(valid_fix2_down.keys()) & set(test_df.index.values)))
print(len(set(valid_fix3_down.keys()) | set(valid_fix_down.keys())))
print(len(set(valid_fix4_down.keys()) & set(test_df.index.values)))
print(len(set(valid_fix6_down.keys()) & set(test_df.index.values)))
# print(len(set(valid_fix2.keys()) & set(test_df.index.values)))
print(len(set(valid_fix_up.keys()) & set(["e30535a390"])))
print(len(set(valid_fix2_up.keys()) & set(["e30535a390"])))
print(len(set(valid_fix_down.keys()) & set(["e30535a390"])))
print(len(set(valid_fix2_down.keys()) & set(["e30535a390"])))
print(len(set(valid_fix3_down.keys()) & set(["e30535a390"])))
print(len(set(valid_fix4_down.keys()) & set(["e30535a390"])))
print(len(set(valid_fix6_down.keys()) & set(["e30535a390"])))
print(len(set(valid_fix_up.keys()) & set(["caf762e1ab"])))
print(len(set(valid_fix2_up.keys()) & set(["caf762e1ab"])))
print(len(set(valid_fix_down.keys()) & set(["caf762e1ab"])))
print(len(set(valid_fix2_down.keys()) & set(["caf762e1ab"])))
print(len(set(valid_fix3_down.keys()) & set(["caf762e1ab"])))
print(len(set(valid_fix4_down.keys()) & set(["caf762e1ab"])))
print(len(set(valid_fix6_down.keys()) & set(["caf762e1ab"])))


# exit()

weights = []
preds = np.zeros((18000,101,101,1))
for i in range(5):
    print(model_count)
    model_count += 1
    unet5_np = np.fromfile("cache/unet8lov_" + str(model_count) + "_np", dtype=np.float32).reshape(-1, img_size_ori,
                                                                                                     img_size_ori, 1)
    preds += (unet5_np*0.5)
    weights.append(0.5)
    unet5_np = np.fromfile("cache/unet8mse_" + str(model_count) + "_np", dtype=np.float32).reshape(-1, img_size_ori,
                                                                                                     img_size_ori, 1)
    preds += (unet5_np*0.5)
    weights.append(0.5)
    unet5_np = np.fromfile("cache/inception2lov_" + str(model_count) + "_np",
                           dtype=np.float32).reshape(-1, img_size_ori,
                                                     img_size_ori, 1)
    preds += (unet5_np*0.2)
    weights.append(0.2)
    unet5_np = np.fromfile("cache/inception2mse_" + str(model_count) + "_np",
                           dtype=np.float32).reshape(-1, img_size_ori,
                                                     img_size_ori, 1)
    preds += (unet5_np*0.2)
    weights.append(0.2)

model_count = 0
for i in range(0,10):
    print(model_count)
    model_count += 1
    unet5_np = np.fromfile("cache/unet10lov_" + str(model_count) + "_np",
                           dtype=np.float32).reshape(-1, img_size_ori,
                                                     img_size_ori, 1)
    preds += (unet5_np * 1.0)
    weights.append(1.0)
    unet5_np = np.fromfile("cache/unet10mse_" + str(model_count) + "_np",
                           dtype=np.float32).reshape(-1, img_size_ori,
                                                     img_size_ori, 1)
    preds += (unet5_np * 1.0)
    weights.append(1.0)
    unet5_np = np.fromfile("cache/unet5lov_" + str(model_count) + "_np",
                           dtype=np.float32).reshape(-1, img_size_ori,
                                                     img_size_ori, 1)
    preds += (unet5_np*0.8)
    weights.append(0.8)
    unet5_np = np.fromfile("cache/unet5mse_" + str(model_count) + "_np",
                           dtype=np.float32).reshape(-1, img_size_ori,
                                                     img_size_ori, 1)
    preds += (unet5_np*0.8)
    weights.append(0.8)
    # unet5_np = np.fromfile("cache/unet15lov_" + str(model_count) + "_np",
    #                        dtype=np.float32).reshape(-1, img_size_ori,
    #                                                  img_size_ori, 1)
    # preds += (unet5_np * 0.6)
    # weights.append(0.6)
    # unet5_np = np.fromfile("cache/unet15mse_" + str(model_count) + "_np",
    #                        dtype=np.float32).reshape(-1, img_size_ori,
    #                                                  img_size_ori, 1)
    # preds += (unet5_np * 0.6)
    # weights.append(0.6)




model_count = 0
test_sample_ids = list(np.load('cache/ids_test.npy'))
# print test_df.index
test_df['index_guido'] = range(test_df.shape[0])
test_df.loc[test_sample_ids,'index_guido'] = range(test_df.shape[0])
# print test_df.loc[:,'index_guido']
for i in range(0,8):
    print(model_count)
    unet5_np = sigmoid(np.load('cache/best_model_{}.npy'.format(model_count))).reshape(-1, img_size_ori, img_size_ori, 1)
    preds += (unet5_np[test_df.loc[:,'index_guido']] * 1.5)
    weights.append(1.5)
    model_count += 1

with open('../feat/puzzle_id.pkl', 'r') as fr:
    processd_id = pkl.load(fr)
preds_puzzle = np.zeros((18000,101,101,1))
weights_puzzle = []
model_count = 0
for i in range(0,10):
    print(model_count)
    model_count += 1
    unet5_np = np.fromfile("cache/unet11lov_" + str(model_count) + "_np",
                           dtype=np.float32).reshape(-1, img_size_ori,
                                                     img_size_ori, 1)
    preds_puzzle += (unet5_np * 1.0)
    weights_puzzle.append(1.0)
    unet5_np = np.fromfile("cache/unet11mse_" + str(model_count) + "_np",
                           dtype=np.float32).reshape(-1, img_size_ori,
                                                     img_size_ori, 1)
    preds_puzzle += (unet5_np * 1.0)
    weights_puzzle.append(1.0)


preds_puzzle = np.array(preds_puzzle/sum(weights_puzzle))
preds = np.array(preds/sum(weights))
print preds.shape,preds_puzzle.shape
preds2 = []


w_puzzle = 2.0
count = 0
for i in range(preds.shape[0]):
    if test_df.index.values[i] in processd_id:
        w_puzzle = 1.5
        count += 1
    else:
        w_puzzle = 0.8
    preds2.append((preds[i] + preds_puzzle[i] * w_puzzle)/(1.0 + w_puzzle))
print(count)

valudcounts = {}
count_up = 0
count_down = 0
count_down2 = 0 # all zero down
count_down4 = 0 # all one down
count_down3 = 0 # vec down infer
count_up2 = 0
count_down6 = 0
for i in range(test_df.index.values.shape[0]):
    # if (test_df.index.values[i]) in valid_fix2.keys():
    #     preds2[i] = valid_fix2[test_df.index.values[i]] + preds2[i] * 0.6
    # if (test_df.index.values[i]) in valid_fix_up.keys() or (test_df.index.values[i]) in valid_fix_down.keys():
    #     preds2[i] *= 0.8
    if (test_df.index.values[i]) in valid_fix_down.keys():
        preds2[i] = valid_fix_down[test_df.index.values[i]] * 0.55  # TODO
        count_down += 1

    elif (test_df.index.values[i]) in valid_fix_up.keys():
        preds2[i] = valid_fix_up[test_df.index.values[i]] * 0.35 + preds2[i] * 0.75
        count_up += 1
    elif (test_df.index.values[i]) in valid_fix3_down.keys():
        preds2[i] = valid_fix3_down[test_df.index.values[i]] * 0.55 + preds2[i] * 0.6
        count_down3 += 1
    else:
        if (test_df.index.values[i]) in valid_fix4_down.keys():
            if valid_fix4_down[test_df.index.values[i]] < 3:
                preds2[i] *= 0.60 # TODO
            else:
                preds2[i] *= 0.75
            count_down4 += 1
        if (test_df.index.values[i]) in valid_fix2_down.keys():
            if valid_fix2_down[test_df.index.values[i]] < 3:
                preds2[i] *= 0.60 # TODO
            else:
                preds2[i] *= 0.75
            count_down2 += 1
        if (test_df.index.values[i]) in valid_fix6_down.keys():
            preds2[i] *= 0.55  # TODO
            if np.sum(np.int32(preds2[i] > index_best[0])) != 0:
                count_down6 += 1

    if (test_df.index.values[i]) in valid_fix2_up.keys() and (test_df.index.values[i]) not in valid_fix6_down.keys() and (test_df.index.values[i]) not in valid_fix2_down.keys():
        preds2[i] += valid_fix2_up[test_df.index.values[i]] * 0.50
        count_up2 += 1
    # if (test_df.index.values[i]) in valid_fix_up.keys() or (test_df.index.values[i]) in valid_fix_down.keys():
    #     preds2[i] *= 0.8
    preds2[i] = np.int32(preds2[i] > index_best[0])
    if np.sum(preds2[i].flatten()) * 1.0 / 10201 < index_best[1]:
        preds2[i][:] = 0.0
    valudcounts[math.ceil(np.sum(preds2[i].flatten()) * 10.0 / 10201)] = valudcounts.get(math.ceil(np.sum(preds2[i].flatten()) * 10.0 / 10201),0) + 1.0

print(count_up,count_down,count_down2,count_down3,count_down4,count_down6,count_up2)
print(map(lambda x:(x[0],x[1]*1.0/18000),valudcounts.items()))
preds2 = np.array(preds2)
preds2.astype(np.float32).tofile("cache/submit2_np")
pred_dict = {idx: RLenc((preds2[i])) for i, idx in
                 enumerate(tqdm_notebook(test_df.index.values))}
sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv("unet_incep_leak_sub14_1.5_0.8.csv")

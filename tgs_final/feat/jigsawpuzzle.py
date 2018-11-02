import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook
from skimage.transform import resize
from PIL import Image
from joblib import delayed, Parallel, dump, load
from keras.preprocessing.image import load_img
import random


img_size_ori = 101
img_size_target = 101
min_length = 15

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def cos(vector1,vector2):
    return np.dot(vector1,vector2)

def dist(vector1,vector2):
    # if type(weight) == type(None):
    #     weight = np.ones(vector1.shape)
    return np.sqrt(np.sum((vector1-vector2)**2))

def imagetran(x):
    x = 2 * (x - 0.5)
    return x

train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df["images"] = [(np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True))/255. )  for idx in
                      (train_df.index)]
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
test_df["images"] = [(np.array(load_img("../input/test/images/{}.png".format(idx), grayscale=True))/255. )  for idx in
                      (test_df.index)]
train_df = train_df.append(test_df)


train_df['top'] = train_df.images.map(lambda x:2 * x[0,:] - x[1,:])
train_df['down'] = train_df.images.map(lambda x:2 * x[-1,:] - x[-2,:])
train_df['left'] = train_df.images.map(lambda x:2 * x[:,0] - x[:,1])
train_df['right'] = train_df.images.map(lambda x:2 * x[:,-1] - x[:,-2])
print train_df['top'].values[0]
train_df['topnorm'] = train_df.top.map(lambda x:(x - np.mean(x))/(np.std(x) + 0.00001))
train_df['downnorm'] = train_df.down.map(lambda x:(x - np.mean(x))/(np.std(x) + 0.00001))
train_df['leftnorm'] = train_df.left.map(lambda x:(x - np.mean(x))/(np.std(x) + 0.00001))
train_df['rightnorm'] = train_df.right.map(lambda x:(x - np.mean(x))/(np.std(x) + 0.00001))
print train_df['topnorm'].values[0]
# train_df['topnorm'] = train_df.topnorm.map(lambda x:x/(np.linalg.norm(x) + 0.0001))
# train_df['downnorm'] = train_df.downnorm.map(lambda x:x/(np.linalg.norm(x) + 0.0001))
# train_df['leftnorm'] = train_df.leftnorm.map(lambda x:x/(np.linalg.norm(x) + 0.0001))
# train_df['rightnorm'] = train_df.rightnorm.map(lambda x:x/(np.linalg.norm(x) + 0.0001))
train_df['topmatch'] = ["0"] * train_df.shape[0]
train_df['downmatch'] = ["0"] * train_df.shape[0]
train_df['leftmatch'] = ["0"] * train_df.shape[0]
train_df['rightmatch'] = ["0"] * train_df.shape[0]


# print(train_df['topnorm'].values[0],train_df['topnorm'].values[1],np.dot(train_df['topnorm'].values[0],train_df['topnorm'].values[1]))
# for i in range(train_df.shape[0]):
#     best_score = 0.0
#     best_score2 = 0.0
#     bestid = "0"
#     bestid2 = "0"
#     v1 = train_df['leftnorm'].values[i]
#     for j in range(train_df.shape[0]):
#         v2 = train_df['rightnorm'].values[j]
#         d = cos(v1,v2)
#         if d > best_score:
#             bestid = train_df.index.values[j]
#             best_score = d
#         elif d > best_score2:
#             bestid2 = train_df.index.values[j]
#             best_score2 = d
#     train_df.loc[train_df.index.values[i], 'leftmatch'] = " ".join([bestid,bestid2,str(best_score),str(best_score2)])
#     # print(" ".join([bestid,bestid2,str(best_score),str(best_score2)]))
#     if i % 100 == 0:
#         print(i)

train_df['topmatch'] = train_df.top.map(lambda x:"")
train_df['downmatch'] = train_df.down.map(lambda x:"")
train_df['leftmatch'] = train_df.left.map(lambda x:"")
train_df['rightmatch'] = train_df.right.map(lambda x:"")
# for i in range(train_df.shape[0]):
#     v1 = train_df.top.values[i]
#     best_score = 100.0
#     best_score2 = 100.0
#     id1 = ""
#     id2 = ""
#     for j in range(train_df.shape[0]):
#         v2 = train_df.down.values[j]
#         d = dist(v1,v2)
#         if d<best_score:
#             best_score = d
#             id1 = train_df.index.values[j]
#         elif d<best_score2:
#             best_score2 = d
#             id2 = train_df.index.values[j]
#     if (best_score2/(best_score + 1e-3) >= 2 or best_score2 - best_score > 5) and best_score < 10:
#         train_df.loc[train_df.index.values[i],'topmatch'] = id1
#     if i% 1000 == 0:
#         print(i)

def proc(i):
    if i % 1000 == 0:
        print(i)
    res = ""
    v1 = train_df.top.values[i]
    best_score = 100.0
    best_score2 = 100.0
    id1 = ""
    id2 = ""
    for j in range(train_df.shape[0]):
        v2 = train_df.down.values[j]
        d = dist(v1, v2)
        if d < best_score:
            best_score = d
            id1 = train_df.index.values[j]
        elif d < best_score2:
            best_score2 = d
            id2 = train_df.index.values[j]
    if (best_score2 / (
        best_score + 1e-5) >= 1.5 or best_score2 - best_score > 5) and best_score < 10:
        res = id1 + " " + str(best_score) + " " + id2 + " " + str(best_score2)
    # print res
    return res

indexs = np.array(range(100))
reslist = Parallel(n_jobs=100)(delayed(proc)(i) for i in indexs)
train_df.loc[train_df.index.values[:100],'topmatch'] = reslist
print(train_df.loc[train_df.index.values[:100],'topmatch'])

def proc(i):
    if i % 1000 == 0:
        print(i)
    res = ""
    v1 = train_df.down.values[i]
    best_score = 100.0
    best_score2 = 100.0
    id1 = ""
    id2 = ""
    for j in range(train_df.shape[0]):
        v2 = train_df.top.values[j]
        d = dist(v1, v2)
        if d < best_score:
            best_score = d
            id1 = train_df.index.values[j]
        elif d < best_score2:
            best_score2 = d
            id2 = train_df.index.values[j]
    if (best_score2 / (
        best_score + 1e-5) >= 2 or best_score2 - best_score > 5) and best_score < 10:
        res = id1 + " " + str(best_score) + " " + id2 + " " + str(best_score)
    return res

indexs = np.array(range(train_df.shape[0]))
reslist = Parallel(n_jobs=100)(delayed(proc)(i) for i in indexs)
train_df.loc[:,'downmatch'] = reslist

def proc(i):
    if i % 1000 == 0:
        print(i)
    res = ""
    v1 = train_df.left.values[i]
    best_score = 100.0
    best_score2 = 100.0
    id1 = ""
    id2 = ""
    for j in range(train_df.shape[0]):
        v2 = train_df.right.values[j]
        d = dist(v1, v2)
        if d < best_score:
            best_score = d
            id1 = train_df.index.values[j]
        elif d < best_score2:
            best_score2 = d
            id2 = train_df.index.values[j]
    if (best_score2 / (
        best_score + 1e-5) >= 2 or best_score2 - best_score > 5) and best_score < 10:
        res = id1 + " " + str(best_score) + " " + id2 + " " + str(best_score)
    return res

indexs = np.array(range(train_df.shape[0]))
reslist = Parallel(n_jobs=100)(delayed(proc)(i) for i in indexs)
train_df.loc[:,'leftmatch'] = reslist

def proc(i):
    if i % 1000 == 0:
        print(i)
    res = ""
    v1 = train_df.right.values[i]
    best_score = 100.0
    best_score2 = 100.0
    id1 = ""
    id2 = ""
    for j in range(train_df.shape[0]):
        v2 = train_df.left.values[j]
        d = dist(v1, v2)
        if d < best_score:
            best_score = d
            id1 = train_df.index.values[j]
        elif d < best_score2:
            best_score2 = d
            id2 = train_df.index.values[j]
    if (best_score2 / (
        best_score + 1e-5) >= 2 or best_score2 - best_score > 5) and best_score < 10:
        res = id1 + " " + str(best_score) + " " + id2 + " " + str(best_score)
    return res

indexs = np.array(range(train_df.shape[0]))
reslist = Parallel(n_jobs=100)(delayed(proc)(i) for i in indexs)
train_df.loc[:,'rightmatch'] = reslist

#
# for i in range(train_df.shape[0]):
#     v1 = train_df.down.values[i]
#     best_score = 100.0
#     best_score2 = 100.0
#     id1 = ""
#     id2 = ""
#     for j in range(train_df.shape[0]):
#         v2 = train_df.top.values[j]
#         d = dist(v1,v2)
#         if d<best_score:
#             best_score = d
#             id1 = train_df.index.values[j]
#         elif d<best_score2:
#             best_score2 = d
#             id2 = train_df.index.values[j]
#     if (best_score2/(best_score + 1e-3) >= 2 or best_score2 - best_score > 5) and best_score < 10:
#         train_df.loc[train_df.index.values[i],'downmatch'] = id1
#     if i% 1000 == 0:
#         print(i)
#
# for i in range(train_df.shape[0]):
#     v1 = train_df.left.values[i]
#     best_score = 100.0
#     best_score2 = 100.0
#     id1 = ""
#     id2 = ""
#     for j in range(train_df.shape[0]):
#         v2 = train_df.right.values[j]
#         d = dist(v1,v2)
#         if d<best_score:
#             best_score = d
#             id1 = train_df.index.values[j]
#         elif d<best_score2:
#             best_score2 = d
#             id2 = train_df.index.values[j]
#     if (best_score2/(best_score + 1e-3) >= 2 or best_score2 - best_score > 5) and best_score < 10:
#         train_df.loc[train_df.index.values[i],'leftmatch'] = id1
#     if i% 1000 == 0:
#         print(i)
#
# for i in range(train_df.shape[0]):
#     v1 = train_df.right.values[i]
#     best_score = 100.0
#     best_score2 = 100.0
#     id1 = ""
#     id2 = ""
#     for j in range(train_df.shape[0]):
#         v2 = train_df.left.values[j]
#         d = dist(v1,v2)
#         if d<best_score:
#             best_score = d
#             id1 = train_df.index.values[j]
#         elif d<best_score2:
#             best_score2 = d
#             id2 = train_df.index.values[j]
#     if (best_score2/(best_score + 1e-3) >= 2 or best_score2 - best_score > 5) and best_score < 15:
#         train_df.loc[train_df.index.values[i],'rightmatch'] = id1
#     if i% 1000 == 0:
#         print(i)



train_df.loc[:,['topmatch','downmatch','leftmatch','rightmatch']].to_csv('puzzleresult2.csv')



import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook
from skimage.transform import resize
from PIL import Image
from joblib import delayed, Parallel, dump, load
from keras.preprocessing.image import load_img,save_img
from skimage.transform import resize
import random
import cPickle as pkl
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")


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

def dist(vector1,vector2):
    return np.sqrt(np.sum((vector1-vector2)**2))

train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df["images"] = [(np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True))/255. )  for idx in
                      (train_df.index)]
train_df["masks"] = [(np.array(
    load_img("../input/train/masks/{}.png".format(idx),
             grayscale=True)) / 255.) for idx in
                      (train_df.index)]
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
test_df["images"] = [(np.array(load_img("../input/test/images/{}.png".format(idx), grayscale=True))/255. )  for idx in
                      (test_df.index)]

train_df = train_df.append(test_df)


train_df['top'] = train_df.images.map(lambda x:1.5 * x[0,:] - 0.5 * x[1,:])
train_df['down'] = train_df.images.map(lambda x:1.5 * x[-1,:] - 0.5 * x[-2,:])
train_df['left'] = train_df.images.map(lambda x:1.5 * x[:,0] - 0.5 * x[:,1])
train_df['right'] = train_df.images.map(lambda x:1.5 * x[:,-1] - 0.5 * x[:,-2])
train_df['top'] = train_df.top.map(lambda x:(x - np.mean(x))/(np.std(x) + 0.00001))
train_df['down'] = train_df.down.map(lambda x:(x - np.mean(x))/(np.std(x) + 0.00001))
train_df['left'] = train_df.left.map(lambda x:(x - np.mean(x))/(np.std(x) + 0.00001))
train_df['right'] = train_df.right.map(lambda x:(x - np.mean(x))/(np.std(x) + 0.00001))

train_df2 = pd.read_csv('puzzleresult2.csv', index_col="id")
train_df = train_df.join(train_df2)

train_df[ 'topmatch'] = train_df[ 'topmatch'].map(str)
train_df[ 'downmatch'] = train_df[ 'downmatch'].map(str)
train_df[ 'leftmatch'] = train_df[ 'leftmatch'].map(str)
train_df[ 'rightmatch'] = train_df[ 'rightmatch'].map(str)

for idx in train_df.index.values:
    if train_df.loc[idx, 'topmatch'].startswith(idx):
        train_df.loc[idx, 'topmatch'] = "nan"
    if train_df.loc[idx, 'downmatch'].startswith(idx):
        train_df.loc[idx, 'downmatch'] = "nan"
    if train_df.loc[idx, 'leftmatch'].startswith(idx):
        train_df.loc[idx, 'leftmatch'] = "nan"
    if train_df.loc[idx, 'rightmatch'].startswith(idx):
        train_df.loc[idx, 'rightmatch'] = "nan"

remain_id = set(train_df.index.values)
valid_id = set(train_df.index.values)
processd_id = set()
results = []
for root in valid_id:
    if root in processd_id:
        continue
    graph_id = np.array([[root]])
    remain_id = set()
    topvalid,downvalid,leftvalid,rightvalid = True,True,True,True
    while(True):
        shape_g = graph_id.shape
        used_id = set(graph_id.flatten())
        if len(remain_id) > 0:
            for i in range(graph_id.shape[0]):
                for j in range(graph_id.shape[1]):
                    if graph_id[i, j] in remain_id:
                        if j > 0 and graph_id[i, j - 1] == "0000000000" and str(
                                train_df.loc[graph_id[i, j], 'leftmatch']) != 'nan':
                            id1 = train_df.loc[graph_id[i, j], 'leftmatch'].split(' ')[0]
                            if id1 in used_id:
                                continue
                            graph_id[i, j - 1] = id1
                            remain_id.add(id1)
                            topvalid, downvalid, leftvalid, rightvalid = True, True, True, True
                        if j + 1 < shape_g[1] and graph_id[i, j + 1] == "0000000000" and str(
                                train_df.loc[graph_id[i, j], 'rightmatch']) != 'nan':
                            id1 = train_df.loc[graph_id[i, j], 'rightmatch'].split(' ')[0]
                            if id1 in used_id:
                                continue
                            graph_id[i, j + 1] = id1
                            remain_id.add(id1)
                            topvalid, downvalid, leftvalid, rightvalid = True, True, True, True
                        if i > 0 and graph_id[i - 1, j] == "0000000000" and str(
                                train_df.loc[graph_id[i, j], 'topmatch']) != 'nan':
                            id1 = train_df.loc[graph_id[i, j], 'topmatch'].split(' ')[0]
                            if id1 in used_id:
                                continue
                            graph_id[i - 1, j] = id1
                            remain_id.add(id1)
                            topvalid, downvalid, leftvalid, rightvalid = True, True, True, True
                        if i + 1 < shape_g[0] and graph_id[i + 1, j] == "0000000000" and str(
                                train_df.loc[graph_id[i, j], 'downmatch']) != 'nan':
                            id1 = train_df.loc[graph_id[i, j], 'downmatch'].split(' ')[0]
                            if id1 in used_id:
                                continue
                            graph_id[i + 1, j] = id1
                            remain_id.add(id1)
                            topvalid, downvalid, leftvalid, rightvalid = True, True, True, True
                        remain_id.remove(graph_id[i, j])
        elif topvalid:
            graph_id_add = np.array(["0000000000"]*shape_g[1]).reshape(1,-1)
            isfind = False
            for j in range(shape_g[1]):
                idtop = graph_id[0,j]
                if idtop != "0000000000" and str(train_df.loc[idtop, 'topmatch']) != 'nan':
                    id1 = train_df.loc[idtop, 'topmatch'].split(' ')[0]
                    id2 = train_df.loc[idtop, 'topmatch'].split(' ')[2]
                    if id1 in used_id:
                        continue
                    graph_id_add[0,j] = id1
                    remain_id.add(id1)
                    isfind = True

                # elif j > 0 and graph_id[0,j-1] != "0000000000" and str(train_df.loc[graph_id[0,j-1], 'rightmatch']) != 'nan':
                #     id1 = train_df.loc[graph_id[0,j-1], 'rightmatch'].split(' ')[0]
                #     graph_id_add[0,j] = id1
            if isfind:
                topvalid, downvalid, leftvalid, rightvalid = True, True, True, True
                graph_id = np.concatenate([graph_id_add,graph_id],axis = 0)
                continue
            else:
                topvalid = False
        elif downvalid:
            graph_id_add = np.array(["0000000000"]*shape_g[1]).reshape(1,-1)
            isfind = False
            for j in range(shape_g[1]):
                idtop = graph_id[-1, j]
                if idtop != "0000000000" and str(train_df.loc[idtop, 'downmatch']) != 'nan':
                    # print(str(train_df.loc[idtop,:]))
                    id1 = train_df.loc[idtop, 'downmatch'].split(' ')[0]
                    id2 = train_df.loc[idtop, 'downmatch'].split(' ')[2]
                    if id1 in used_id:
                        continue
                    graph_id_add[0,j] = id1
                    remain_id.add(id1)
                    isfind = True
                # elif j > 0 and graph_id[-1, j - 1] != "0000000000" and str(
                #     train_df.loc[graph_id[-1, j - 1], 'rightmatch']) != 'nan':
                #     id1 = train_df.loc[graph_id[-1, j - 1], 'rightmatch'].split(' ')[0]
                #     graph_id_add[0,j] = id1
            if isfind:
                topvalid, downvalid, leftvalid, rightvalid = True, True, True, True
                graph_id = np.concatenate([graph_id, graph_id_add], axis=0)
                continue
            else:
                downvalid = False

        elif leftvalid:
            graph_id_add = np.array(["0000000000"] * shape_g[0]).reshape(-1,1)
            isfind = False
            for j in range(shape_g[0]):
                idtop = graph_id[j,0]
                if idtop != "0000000000" and str(train_df.loc[idtop, 'leftmatch']) != 'nan':
                    id1 = train_df.loc[idtop, 'leftmatch'].split(' ')[0]
                    id2 = train_df.loc[idtop, 'leftmatch'].split(' ')[2]
                    if id1 in used_id:
                        continue
                    graph_id_add[j,0] = id1
                    remain_id.add(id1)
                    isfind = True
                # elif j > 0 and graph_id[j-1, 0] != "0000000000" and str(
                #     train_df.loc[graph_id[j-1, 0], 'downmatch']) != 'nan':
                #     id1 = train_df.loc[graph_id[j-1, 0], 'downmatch'].split(' ')[0]
                #     graph_id_add[j,0] = id1
            if isfind:
                topvalid, downvalid, leftvalid, rightvalid = True, True, True, True
                graph_id = np.concatenate([graph_id_add, graph_id], axis=1)
                continue
            else:
                leftvalid = False

        elif rightvalid:
            graph_id_add = np.array(["0000000000"] * shape_g[0]).reshape(-1,1)
            isfind = False
            for j in range(shape_g[0]):
                idtop = graph_id[j,-1]
                if idtop != "0000000000" and str(train_df.loc[idtop, 'rightmatch']) != 'nan':
                    id1 = train_df.loc[idtop, 'rightmatch'].split(' ')[0]
                    id2 = train_df.loc[idtop, 'rightmatch'].split(' ')[2]
                    if id1 in used_id:
                        continue
                    graph_id_add[j,0] = id1
                    remain_id.add(id1)
                    isfind = True
                # elif j > 0 and graph_id[j - 1, -1] != "0000000000" and str(
                #         train_df.loc[graph_id[j - 1, -1], 'downmatch']) != 'nan':
                #     id1 = \
                #     train_df.loc[graph_id[j - 1, -1], 'downmatch'].split(' ')[0]
                #     graph_id_add[j, 0] = id1
            if isfind:
                topvalid, downvalid, leftvalid, rightvalid = True, True, True, True
                graph_id = np.concatenate([graph_id, graph_id_add], axis=1)
                continue
            else:
                rightvalid = False
        else:
            break

    graph_id = np.concatenate([graph_id, np.array(["0000000000"] * shape_g[0]).reshape(-1,1)], axis=1)
    graph_id = np.concatenate([np.array(["0000000000"] * shape_g[0]).reshape(-1, 1),graph_id],
        axis=1)
    shape_g = graph_id.shape
    graph_id = np.concatenate(
        [graph_id, np.array(["0000000000"] * shape_g[1]).reshape(1, -1)],
        axis=0)
    graph_id = np.concatenate(
        [np.array(["0000000000"] * shape_g[1]).reshape(1, -1),graph_id],
        axis=0)

    # check & aug
    for i in range(graph_id.shape[0]):
        for j in range(graph_id.shape[1]):
            if graph_id[i, j] != "0000000000":
                score = 0.0
                num = 0
                if graph_id[i, j - 1] != "0000000000":
                    score += 8 - dist(train_df.loc[graph_id[i, j],'left'], train_df.loc[graph_id[i, j - 1],'right'])
                    num += 1
                if graph_id[i, j + 1] != "0000000000":
                    score += 8 - dist(train_df.loc[graph_id[i, j], 'right'],
                                 train_df.loc[graph_id[i, j + 1], 'left'])
                    num += 1
                if graph_id[i - 1, j] != "0000000000":
                    score += 8 - dist(train_df.loc[graph_id[i, j], 'top'],
                                     train_df.loc[graph_id[i - 1, j], 'down'])
                    num +=1
                if graph_id[i + 1, j] != "0000000000":
                    score += 8 - dist(train_df.loc[graph_id[i, j], 'down'],
                                         train_df.loc[
                                             graph_id[i + 1, j], 'top'])
                    num += 1
                if (score < -2.0 and num > 1) or (score < -2.0):
                    used_id.remove(graph_id[i, j])
                    print("find error:" + graph_id[i, j])
                    graph_id[i, j] = "0000000000"
    # plot
    # print graph_id
    if graph_id.shape[0] > 4 or graph_id.shape[1] > 4:
        processd_id = processd_id | used_id
        graph = np.zeros((graph_id.shape[0] * 101,graph_id.shape[1] * 101,1)) + 0.
        masks = np.zeros((graph_id.shape[0] * 101,graph_id.shape[1] * 101,1)) + 0.
        for i in range(graph_id.shape[0]):
            for j in range(graph_id.shape[1]):
                if graph_id[i,j] != "0000000000":
                    graph[i*101:(i+1)*101,j*101:(j+1)*101,:] = np.array(train_df.loc[graph_id[i,j],"images"]).reshape(101,101,1)
                    if graph_id[i,j] not in test_df.index:
                        masks[i * 101:(i + 1) * 101, j * 101:(j + 1) * 101, :] = np.array(
            train_df.loc[graph_id[i, j], "masks"]).reshape(101, 101, 1)
        # print graph_id
        save_img("./debug/{}.png".format(root),
                 graph * 255,
                 grayscale=True)
        padsize1 = 13 + 32
        padsize2 = 14 + 32
        padsize1 = 13 + 16
        padsize2 = 14 + 16
        # padsize1 = 101
        # padsize2 = 101
        for i in range(graph_id.shape[0]):
            for j in range(graph_id.shape[1]):
                if graph_id[i, j] != "0000000000":
                    aug_images = (graph[i * 101- padsize1:(i + 1) * 101 + padsize2, j * 101- padsize1:(j + 1) * 101 + padsize2, :] * 255).copy()
                    save_img("../input/aug_images3/{}.png".format(graph_id[i, j]), aug_images,
                             grayscale=True, scale=False)
                    padsize1 = 0
                    padsize2 = 0
                    aug_mask = (masks[i * 101- padsize1:(i + 1) * 101 + padsize2, j * 101- padsize1:(j + 1) * 101 + padsize2, :] * 255).copy()
                    aug_mask[padsize1:-padsize2, padsize1:-padsize2, :] = 0.5 * 255
                    aug_mask[:, :, :] = 0.5 * 255
                    aug_mask[0, :, :] = masks[i * 101 - 1, j * 101- padsize1:(j + 1) * 101 + padsize2, :] * 255
                    aug_mask[-1, :, :] = masks[(i + 1) * 101, j * 101- padsize1:(j + 1) * 101 + padsize2, :] * 255
                    aug_mask[ :,0, :] = masks[i * 101- padsize1:(i + 1) * 101 + padsize2, j * 101 - 1, :] * 255
                    aug_mask[:, -1, :] = masks[i * 101- padsize1:(i + 1) * 101 + padsize2,(j + 1) * 101,  :] * 255
                    save_img("../input/aug_masks3/{}.png".format(graph_id[i, j]), aug_mask,
                             grayscale=True, scale=False)
                    padsize1 = 13 + 32
                    padsize2 = 14 + 32
                    aug_mask = (masks[i * 101- padsize1:(i + 1) * 101 + padsize2, j * 101- padsize1:(j + 1) * 101 + padsize2, :] * 255).copy()
                    aug_mask[padsize1:-padsize2, padsize1:-padsize2, :] = 0.5 * 255
                    save_img("../input/aug_masks2/{}.png".format(graph_id[i, j]), aug_mask,
                             grayscale=True, scale=False)
        results.append((graph_id.shape,list(graph_id.flatten())))


for root in valid_id - processd_id:
    graph = np.zeros((3 * 101, 3 * 101, 1)) + 0.
    masks = np.zeros((3 * 101, 3 * 101, 1)) + 0.5
    padsize1 = 13 + 32
    padsize2 = 14 + 32
    padsize1 = 13 + 16
    padsize2 = 14 + 16
    i = 1
    j = 1
    graph[i * 101:(i + 1) * 101, j * 101:(j + 1) * 101, :] = np.array(
        train_df.loc[root, "images"]).reshape(101, 101, 1)
    save_img("../input/aug_images3/{}.png".format(root),
             graph[i * 101 - padsize1:(i + 1) * 101 + padsize2,
             j * 101 - padsize1:(j + 1) * 101 + padsize2, :] * 255,
             grayscale=True, scale=False)
    padsize1 = 0
    padsize2 = 0
    aug_mask = (masks[i * 101 - padsize1:(i + 1) * 101 + padsize2,
                j * 101 - padsize1:(j + 1) * 101 + padsize2, :] * 255)
    save_img("../input/aug_masks3/{}.png".format(root), aug_mask,
             grayscale=True, scale=False)
    padsize1 = 13 + 32
    padsize2 = 14 + 32
    aug_mask = (masks[i * 101 - padsize1:(i + 1) * 101 + padsize2,
                j * 101 - padsize1:(j + 1) * 101 + padsize2, :] * 255)
    save_img("../input/aug_masks2/{}.png".format(root), aug_mask,
             grayscale=True, scale=False)
with open('./puzzle_id.pkl', 'w') as fw:
    pkl.dump(processd_id, fw)

print(len(processd_id))
import json
results = [json.dumps(_) for _ in results]
with open('./puzzle.pkl', 'w') as fw:
    pkl.dump(results, fw)

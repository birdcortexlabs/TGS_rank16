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
import math,gc
from util import *
from keras.optimizers import *
import random,sys
from sklearn.model_selection import StratifiedKFold



img_size_ori = 101
img_size_target = 128

def mycustom2(y_true, y_pred):
    y_weight = 1.0/K.clip(y_true,1e-2,1)
    return K.mean(K.square(K.abs(y_pred - y_true)), axis=-1) * K.sqrt(y_weight)

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


def imagetran(x):
    x = 4 * (x - 0.5)
    return x

train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
train_df["images"] = [imagetran(np.array(load_img("../input/train/images/{}.png".format(idx), grayscale=True)) / 255.0)  for idx in
                      tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255.0 for idx in
                     tqdm_notebook(train_df.index)]
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
train_df["isempty"] = train_df.coverage.map(lambda x:int(x==0))
x_test = np.array(
    [imagetran(upsample(np.array(load_img("../input/test/images/{}.png".format(idx),
                                grayscale=True)) / 255.0)) for idx in
     tqdm_notebook(test_df.index)]).reshape(-1, img_size_target,
                                            img_size_target, 1)


def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

#
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)






def build_model(start_neurons):
    parms = [
        ('c5_1', ((3, 3, 10 * start_neurons, 16 * start_neurons), 16 * start_neurons)),
        ('c5_2', ((3, 3, 16 * start_neurons, 16 * start_neurons), 16 * start_neurons)),
        ('c5_3', ((3, 3, 16 * start_neurons, 16 * start_neurons), 16 * start_neurons))
    ]

    input_layer = Input((img_size_target, img_size_target, 1), dtype='float32')
    input2 = Input(shape=(1,), dtype='float32')
    x2 = BatchNormalization()(input2)
    x2 = input2

    x = Conv2D(64, kernel_size=7, strides=1)(input_layer)
    x = Conv2D(128, kernel_size=3, strides=2)(x)
    # x = Conv2D(64, kernel_size=3, strides=2)(x)
    xc1 = Mean_std()(x)
    xc2 = Mean_std()(x)
    x = xc1
    hidden = concatenate(
        [Dense(48, activation='relu')(x), Dense(8, activation='sigmoid')(x2),
         Dense(8, activation='relu')(x2)])
    hidden = Dropout(0.05)(hidden)
    meta_output = []
    for name, shape in parms:
        hidden2 = concatenate([Dense(16, activation='sigmoid')(x),
                               Dense(16, activation='sigmoid')(x2),
                               Dense(16, activation='relu')(x)])
        hidden2 = concatenate([Dropout(0.05)(hidden2), hidden])
        kernel = Dense(np.prod(shape[0]), activation='linear')(hidden2)
        bias = Dense(shape[1], activation='linear')(hidden2)
        meta_output.append(concatenate([Lambda(lambda x: 0.005 * x)(kernel),
                                        Lambda(lambda x: 0.005 * x)(bias)]))

    def xception_unit(x, num):
        res = []
        res.append(Conv2D(num, (3, 3), activation="relu",
                          padding="same")(x))
        res.append(Conv2D(num, (1, 1), activation="relu",
                          padding="same")(Conv2D(num, (3, 3), activation="relu",
                                                 padding="same")(x)))
        res.append(Conv2D(num, (3, 3), activation="relu",
                          padding="same")(Conv2D(num, (3, 3), activation="relu",
                                                 padding="same")(x)))
        res.append(Conv2D(num, (4, 4), activation="relu",
                          padding="same")(x))
        res.append(Conv2D(num, (1, 1), activation="relu",
                          padding="same")(Conv2D(num, (1, 4), activation="relu",
                                                 padding="same")(Conv2D(num, (4, 1), activation="relu",
                                                                        padding="same")(x))))
        return concatenate(res)

    i = 0
    # 128 -> 64
    conv1 = xception_unit(input_layer, start_neurons)
    conv1 = Conv2D(start_neurons, (3, 3), activation="relu",
                   padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = xception_unit(conv1, start_neurons)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = xception_unit(pool1, start_neurons)
    conv2 = Conv2D(start_neurons * 4, (3, 3), activation="relu",
                   padding="same")(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = xception_unit(conv2, start_neurons)
    conv2 = BatchNormalization()(conv2)
    # conv2 = concatenate([conv2,pool1])
    conv2 = Conv2DInfer(activation='relu')(
        [conv2, BatchNormalization()(Mean_std()(conv2))])
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.4)(pool2)

    # 32 -> 16
    conv3 = xception_unit(pool2, start_neurons * 2)
    conv3 = BatchNormalization()(conv3)
    conv3 = xception_unit(conv3, start_neurons * 2)
    conv3 = BatchNormalization()(conv3)
    # conv3 = Conv2DInfer(activation='relu')([conv3, concatenate([hidden, BatchNormalization()(Mean_std()(conv3))])])
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.4)(pool3)

    # 16 -> 8
    conv4 = xception_unit(pool3, start_neurons * 3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 10, (3, 3), activation="relu",
                   padding="same")(conv4)
    # conv4 = Conv2D(start_neurons * 4, (1, 1), activation="relu",
    #                padding="same")(conv4)
    conv4 = Conv2D(start_neurons * 10, (3, 3), activation="relu",
                   padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2DInfer(activation='relu')(
        [conv4, BatchNormalization()(Mean_std()(conv4))])
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.4)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu",
                     padding="same")(pool4)
    convm = BatchNormalization()(convm)
    i += 1
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu",
                     padding="same")(convm)
    convm = BatchNormalization()(convm)
    # convm  = concatenate([convm,pool4])

    x = Dense(32,activation='relu')(concatenate([Flatten()(convm),Mean_std()(convm)]))
    output_coverage = Dense(1,activation='relu',name = 'coverage_output')(Dropout(0.4)(x))

    model = Model([input_layer], output_coverage)
    return model

preds = []
ids = []
preds_test = []
ids_test = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_count = 0
data = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
labels = np.array(map(lambda x: math.log(x + 1, 10), train_df.z.values))

is_train = False
for train_idx, val_idx in skf.split(data, train_df.coverage_class):
    if is_train and model_count != int(sys.argv[1]):
        model_count+=1
        continue
    model = build_model(8)
    model.compile(loss='mse', optimizer="adadelta", metrics=["accuracy","mse"])
    ids_train, ids_valid,x_train, x_valid, y_train, y_valid, depth_train, depth_test = \
        train_df.index.values[train_idx],train_df.index.values[val_idx], \
        data[train_idx], data[val_idx], \
        labels[train_idx], labels[val_idx], \
        train_df.z.values[train_idx],train_df.z.values[val_idx]
    depth_train = np.array(map(lambda x: math.log(x + 1, 10), depth_train))
    depth_test = np.array(map(lambda x: math.log(x + 1, 10), depth_test))
    sample_weight = [1.0] * len(x_train)
    x_train_org = x_train.copy()
    y_train_org = y_train.copy()
    depth_train_org = depth_train.copy()
    x_train = np.append(x_train_org, [np.fliplr(x) for x in x_train_org], axis=0)
    y_train = np.append(y_train_org, y_train_org, axis=0)
    depth_train = np.append(depth_train_org, np.array(map(lambda x:x ,depth_train_org)), axis=0)
    sample_weight += [0.6] * len(x_train_org)

    early_stopping = EarlyStopping(patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint("cache/depth_"+str(model_count)+".h5",save_weights_only=True, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.15, patience=6, min_lr=0.00002, verbose=1)
    epochs = 100
    batch_size = 32
    if is_train:
        history = model.fit([x_train], y_train,
                            validation_data=[[x_valid], y_valid],
                            epochs=30,
                            batch_size=40,
                            sample_weight=np.array(sample_weight))
        model.compile(loss='mse', optimizer="adadelta", metrics=["accuracy", "mse"])
        K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * 0.6)
        history = model.fit([x_train], y_train,
                            validation_data=[[x_valid], y_valid],
                            epochs=epochs,
                            batch_size=batch_size,
                            sample_weight = np.array(sample_weight),
                            callbacks=[early_stopping, model_checkpoint, reduce_lr])


    model.load_weights("cache/depth_"+str(model_count)+".h5")
    res = model.predict([x_valid])
    for pid, p in zip(ids_valid, res):
        preds.append(p)
        ids.append(pid)
    res = model.predict([x_test])
    for pid, p in zip(test_df.index, res):
        preds_test.append(p)
        ids_test.append(pid)

    model_count+=1
    del model
    gc.collect()

if not is_train:
    pd.DataFrame({"id":np.array(ids).flatten(),"depthm":np.array(preds).flatten()}).to_csv('./f4_for_stacking.csv',index=None)
    df = pd.DataFrame({"id": np.array(ids_test).flatten(), "depthm": np.array(preds_test).flatten()}).groupby("id").mean()
    df.reset_index(inplace = True)
    df.to_csv('./f4_for_stacking_test.csv',
            index=None)
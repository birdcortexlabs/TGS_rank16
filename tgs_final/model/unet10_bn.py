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
from llose import *
from keras.optimizers import *
import random
import keras
from keras.losses import binary_crossentropy
from scipy.misc import imsave
from sklearn.model_selection import StratifiedKFold
import sys,gc

from keras.utils import multi_gpu_model


img_size_ori = 101
img_size_target = 128

def mycustom(y_true, y_pred):
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    y_union = K.clip(y_true+y_pred,1e-5,1)
    y_weight = K.clip(img_size_target * img_size_target * 1.0 / K.sum(y_union, axis=-1), 1.0, 80.0)
    return K.mean(K.square(K.abs(y_pred - y_true)+0.15), axis=-1) * y_weight

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def mycustom2(y_true, y_pred):
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    return K.mean(lovasz_hinge(y_pred,y_true))

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

def upsample2(img):
    if img_size_ori == img_size_target:
        return img
    if img.shape[0] == img_size_target:
        return img
    img = img.reshape((img.shape[0], img.shape[1]))
    return np.pad(img,((13,14),(13,14)),mode ='constant',constant_values=((0,0),(0,0)))


def downsample2(img):
    if img_size_ori == img_size_target:
        return img
    if img.shape[0] == img_size_ori:
        return img
    return img[13:img.shape[0]-14,13:img.shape[0]-14]

def imagetran(x):
    x = 2 * (x - 0.5)
    return x


train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv("../input/depths.csv", index_col="id")
feat_df = pd.read_csv("./f1_for_stacking.csv", index_col="id")
feat_df = feat_df.join(pd.read_csv("./f2_for_stacking.csv", index_col="id"))
feat_df = feat_df.join(pd.read_csv("./f4_for_stacking.csv", index_col="id"))
train_df = train_df.join(depths_df)
train_df = train_df.join(feat_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]
feat_df = pd.read_csv("./f1_for_stacking_test.csv", index_col="id")
feat_df = feat_df.join(pd.read_csv("./f2_for_stacking_test.csv", index_col="id"))
feat_df = feat_df.join(pd.read_csv("./f4_for_stacking_test.csv", index_col="id"))
test_df = test_df.join(feat_df)
train_df["images"] = [(imagetran(np.array(
    load_img("../input/train/images/{}.png".format(idx),
             grayscale=True)) / 255.0)) for idx in
                      tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img("../input/train/masks/{}.png".format(idx), grayscale=True)) / 255.0 for idx in
                     tqdm_notebook(train_df.index)]
train_df["magic1"] = train_df.masks.map(lambda x:np.mean(np.abs(np.std(x.reshape(img_size_ori,img_size_ori),axis = 0)))<=0.005 and np.sum(x) >= img_size_ori*img_size_ori*0.1 and np.sum(x) <= img_size_ori*img_size_ori*0.8)
train_df["weights"] = train_df.magic1.map(lambda x:0.8 if x else 1.0)
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

train_df["weights"] = train_df["weights"] * train_df["coverage"].map(lambda x: 1.2 if x< 0.15 else 1.0)
# train_df["isempty"] = train_df.coverage.map(lambda x:int(x==0))
y_org = np.array(train_df["masks"].values.tolist()).reshape(-1, img_size_ori, img_size_ori, 1)
train_df["z"] = train_df["z"].map(lambda x:math.log(x+1,10))
test_df["z"] = test_df["z"].map(lambda x:math.log(x+1,10))
x_test = np.array(
    [upsample(imagetran(np.array(load_img("../input/test/images/{}.png".format(idx),
                                grayscale=True)) / 255.0)) for idx in
     tqdm_notebook(test_df.index)]).reshape(-1, img_size_target,
                                            img_size_target, 1)
depth_test = test_df.loc[:, ['z', 'isempty', 'cover']].values

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

#
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)




def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
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

class LossHistory(keras.callbacks.Callback):
    def __init__(self, X_val, y_val, model,thre= 0.5):
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.thre = thre

    def on_epoch_end(self, epoch, logs={}):
        thres = self.thre
        length = self.X_val[0].shape[0] / 3

        preds_valid = self.model.predict(self.X_val)
        preds_valid = np.array([downsample(x) for x in preds_valid])
        y_valid = np.array(self.y_val)
        # print(preds_valid.shape,y_valid.shape)
        iou2 = iou_metric_batch(np.int32(y_valid > 0.5), np.int32(preds_valid[:length] > thres),0.005)
        # preds_valid = (preds_valid[:length] + np.array([np.flipud(x) for x in preds_valid[length:]]) * 0.4)/1.4
        preds_valid2 = (preds_valid[:length] + np.array([np.fliplr(x) for x in preds_valid[length:2*length]]) * 0.6)/1.6
        iou = iou_metric_batch(np.int32(y_valid > 0.5), np.int32(preds_valid2 > thres),0.005)
        logs['iou'] = iou
        preds_valid = (preds_valid[:length] + np.array([np.fliplr(x) for x in preds_valid[length:2*length]]) * 0.6 +
                       + np.array([downsample2(upsample(x)).reshape((img_size_ori, img_size_ori, 1)) for x in preds_valid[length * 2:]]) * 0.6
                       )/2.2
        iou3 = iou_metric_batch(np.int32(y_valid > 0.5), np.int32(preds_valid > thres),0.005)
        print('val iou:{},iouorg:{},iou3:{},mean:{}'.format(iou,iou2,iou3,np.mean(preds_valid)))
        del preds_valid
        gc.collect()



def build_model(start_neurons):
    parms = [
        ('c5_1',
         ((3, 3, 10 * start_neurons, 16 * start_neurons), 16 * start_neurons)),
        ('c5_2',
         ((3, 3, 16 * start_neurons, 16 * start_neurons), 16 * start_neurons)),
        ('c5_3',
         ((3, 3, 16 * start_neurons, 16 * start_neurons), 16 * start_neurons))
    ]

    input_layer = Input((img_size_target, img_size_target, 1), dtype='float32')
    input2 = Input(shape=(3,), dtype='float32')
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
        meta_output.append(concatenate([Lambda(lambda x: 0.25 * x)(kernel),
                                        Lambda(lambda x: 0.05 * x)(bias)]))

    def xception_unit(x, num):
        res = []
        # x = Add()([Conv2D(int(x.shape[3]), (3, 3), activation="relu",
        #            padding="same")(x),x])
        # res.append(x)
        res.append(Conv2D(num * 2, (3, 3), activation="relu",
                          padding="same")(x))
        res.append(Conv2D(num * 2, (3, 3), activation="relu",
                          padding="same")(
            Conv2D(num * 2, (3, 3), activation="relu",
                   padding="same")(x)))
        res.append(Conv2D(num * 2, (3, 3), activation="relu",
                          padding="same")(x))
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
    conv2 = concatenate([conv2, pool1])
    conv2 = BatchNormalization()(conv2)
    conv2 = xception_unit(conv2, start_neurons)
    conv2 = BatchNormalization()(conv2)
    # conv2 = concatenate([conv2,pool1])
    conv2 = Conv2DInfer(activation='relu')(
        [conv2, concatenate([hidden, BatchNormalization()(Mean_std()(conv2))])])
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.4)(pool2)

    # 32 -> 16
    conv3 = xception_unit(pool2, start_neurons * 2)
    # conv3re = conv3
    conv3 = concatenate([conv3, pool2])
    conv3 = BatchNormalization()(conv3)
    conv3 = xception_unit(conv3, start_neurons * 2)
    # conv3 = Add()([conv3re,conv3])
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.4)(pool3)

    # 16 -> 8
    conv4 = xception_unit(pool3, start_neurons * 3)
    # conv4 = Conv2D(start_neurons * 10, (3, 3), activation="relu",
    #                padding="same")(pool3)
    conv4 = concatenate([conv4, pool3])
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(start_neurons * 10, (3, 3), activation="relu",
                   padding="same")(conv4)
    conv4 = Conv2D(start_neurons * 10, (3, 3), activation="relu",
                   padding="same")(conv4)
    # conv4 = Add()([conv4,Conv2D(start_neurons * 10, (3, 3), activation="relu",
    #                padding="same")(conv4)])
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2DInfer(activation='relu')(
        [conv4, concatenate([hidden, BatchNormalization()(Mean_std()(conv4))])])
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.4)(pool4)

    # Middle
    # convm = MyConv2D(start_neurons * 16, (3, 3), activation="relu",
    #                  padding="same", shape=parms[i])([pool4, meta_output[i]])
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu",
                   padding="same")(pool4)
    convm = BatchNormalization()(convm)
    i += 1
    # convm = MyConv2D(start_neurons * 16, (3, 3), activation="relu",
    #                  padding="same", shape=parms[i])([convm, meta_output[i]])
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu",
                   padding="same")(convm)
    convm = BatchNormalization()(convm)
    # i += 1
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu",
                   padding="same")(convm)
    # convm = BatchNormalization()(convm)
    convm = Conv2D(start_neurons * 16, (3, 3), activation="relu",
                   padding="same")(convm)
    # convm = BatchNormalization()(convm)
    convm = Dropout(0.3)(convm)

    # convm2 = Flatten()(pool4)
    # convm2 = concatenate([Dense(16,activation="relu")(convm2),Dense(16,activation="sigmoid")(convm2)])
    # convm2 = Dense(8 * 8 * 8,activation="relu")(convm2)
    # convm2 = Reshape((8,8,8))(convm2)
    # convm = concatenate([convm,convm2])



    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 6, (3, 3), strides=(2, 2),
                              padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.4)(uconv4)
    u4 = uconv4
    uconv4 = Conv2D(start_neurons * 6, (3, 3), activation="relu",
                    padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Conv2D(start_neurons * 6, (3, 3), activation="relu",
                    padding="same")(uconv4)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = concatenate([uconv4, u4])
    # uconv4 = Conv2DInfer(activation='sigmoid')([uconv4, concatenate([hidden, BatchNormalization()(Mean_std()(uconv4))])])

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2),
                              padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.4)(uconv3)
    # uconv3 = Conv2D(start_neurons * 2, (1, 1), padding="same",
    #                 activation="relu")(uconv3)
    u3 = uconv3
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu",
                    padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu",
                    padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = concatenate([uconv3, u3])

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2),
                              padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.3)(uconv2)
    u2 = uconv2
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation="relu",
                    padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu",
                    padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = concatenate([uconv2, u2])

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2),
                              padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.3)(uconv1)
    u1 = uconv1
    uconv1 = Conv2D(start_neurons * 4, (3, 3), activation="relu",
                    padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Conv2D(start_neurons * 4, (3, 3), activation="relu",
                    padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = concatenate([uconv1, u1])
    # uconv1 = Conv2DInfer(activation='sigmoid')([uconv1, concatenate([hidden, BatchNormalization()(Mean_std()(uconv1))])])

    middle2 = uconv1
    uconv1 = Dropout(0.3)(uconv1)
    uconv1 = Conv2D(16, (1, 1), padding="same", activation="relu")(uconv1)
    uconv1 = Conv2D(16, (1, 1), padding="same", activation="relu")(uconv1)

    pretrain_model = Model([input_layer, input2], pool2)
    output_layer = Conv2D(1, (1, 1), padding="same",
                          name='main_output')(uconv1)
    model2 = Model([input_layer, input2], output_layer)
    output_layer = Lambda(lambda x:K.sigmoid(x))(output_layer)
    model = Model([input_layer, input2], output_layer)
    # model = multi_gpu_model(model, 2)
    # model2 = multi_gpu_model(model2, 2)
    return model,model2,pretrain_model



model_count = 0
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
images = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
masks = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
labels = train_df.coverage_class
for train_idx, val_idx in skf.split(images, labels):
    print(model_count)
    if model_count != int(sys.argv[1]):
        model_count+=1
        continue
    model_count += 1
    ids_train, ids_valid, x_train, x_valid, y_train, y_valid, depth_train,\
    depth_valid,y_train_org, y_valid_org = \
        train_df.index.values[train_idx], train_df.index.values[val_idx], \
        images[train_idx], images[val_idx], \
        masks[train_idx], masks[val_idx], \
        train_df.loc[:, ['z', 'isempty','cover']].values[train_idx], train_df.loc[:,['z','isempty','cover']].values[val_idx], \
        y_org[train_idx],y_org[val_idx]

    weights_train = train_df.weights.values[train_idx]
    model,model2,pretrain_model = build_model(14)
    model.compile(loss=mycustom, optimizer=RMSprop(lr=0.0012),
                  metrics=["accuracy", "mse"])
    model2.compile(loss=mycustom2, optimizer=RMSprop(lr=0.0005))
    sample_weight = weights_train.tolist()

    x_train_org = x_train.copy()
    y_train_org = y_train.copy()
    depth_train_org = depth_train.copy()
    x_train = np.append(x_train, [np.fliplr(x) for x in x_train_org], axis=0)
    y_train = np.append(y_train, [np.fliplr(x) for x in y_train_org], axis=0)
    depth_train = np.append(depth_train, np.array(map(lambda x:x ,depth_train_org)), axis=0)
    sample_weight += (0.8 * weights_train).tolist()

    x_train = np.append(x_train,
                        [upsample2(downsample(x)).reshape((img_size_target, img_size_target, 1)) for x in x_train_org],
                        axis=0).astype(np.float32)
    y_train = np.append(y_train,
                        [upsample2(downsample(x)).reshape((img_size_target, img_size_target, 1)) for x in y_train_org],
                        axis=0).astype(np.float32)
    depth_train = np.append(depth_train, np.array(map(lambda x: x, depth_train_org)), axis=0).astype(np.float32)
    sample_weight += (0.6 * weights_train).tolist()

    x_valid_org = x_valid.copy()
    y_valid_org_temp = y_valid.copy()
    depth_valid_org = depth_valid.copy()

    x_valid2 = np.append(x_valid, [np.fliplr(x) for x in x_valid_org], axis=0)
    y_valid2 = np.append(y_valid, [np.fliplr(x) for x in y_valid_org_temp], axis=0)
    depth_valid2 = np.append(depth_valid, np.array(map(lambda x: x, depth_valid_org)), axis=0)

    x_valid2 = np.append(x_valid2,
                         [upsample2(downsample(x)).reshape((img_size_target, img_size_target, 1)) for x in x_valid_org],
                         axis=0).astype(np.float32)
    y_valid2 = np.append(y_valid2, [upsample2(downsample(x)).reshape((img_size_target, img_size_target, 1)) for x in
                                    y_valid_org_temp], axis=0).astype(np.float32)
    depth_valid2 = np.append(depth_valid2, np.array(map(lambda x: x, depth_valid_org)), axis=0).astype(np.float32)

    lh = LossHistory([x_valid2,depth_valid2], y_valid_org, model)
    early_stopping = EarlyStopping(patience=15, verbose=1)
    modelname = 'cache/unet10_'+str(model_count) + '.h5'
    model_checkpoint = ModelCheckpoint(modelname,save_weights_only=True, save_best_only=True, verbose=1, mode='max',monitor='iou')
    reduce_lr = ReduceLROnPlateau(factor=0.15, patience=8, min_lr=0.00005, verbose=1)
    epochs = 250
    batch_size = 32

    del x_train_org,y_train_org
    history = model.fit([x_train,depth_train], y_train,
                        validation_data=[[x_valid2,depth_valid2], y_valid2],
                        epochs=200,
                        batch_size=40,
                        sample_weight = np.array(sample_weight),
                        callbacks=[lh])
    K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr) * 0.8)
    # model.load_weights(modelname)
    history = model.fit([x_train,depth_train], y_train,
                        validation_data=[[x_valid2,depth_valid2], y_valid2],
                        epochs=80,
                        batch_size=30,
                        sample_weight = np.array(sample_weight),
                        callbacks=[lh,model_checkpoint])
    model.load_weights(modelname)
    model.compile(loss=mycustom, optimizer=RMSprop(lr=0.0008),
                  metrics=["accuracy", "mse"])
    history = model.fit([x_train, depth_train], y_train,
                        validation_data=[[x_valid2, depth_valid2], y_valid2],
                        epochs=epochs,
                        batch_size=batch_size,
                        sample_weight=np.array(sample_weight),
                        callbacks=[early_stopping, reduce_lr, lh,model_checkpoint])
    model.load_weights(modelname)

    preds_test = model.predict([x_test, depth_test], batch_size=40)

    semi_count = 12000
    random.seed(model_count)
    index_test = random.sample(range(preds_test.shape[0]), semi_count)
    x_train = np.append(x_train, x_test[index_test], axis=0)
    y_train = np.append(y_train, np.where(preds_test > 0.5, preds_test,
                                          np.zeros(preds_test.shape))[
        index_test], axis=0)
    depth_train = np.append(depth_train, depth_test[index_test], axis=0)
    sample_weight_test = np.where(depth_test[index_test, 2] > 0.6, 0.4, 0.1)
    sample_weight = sample_weight + sample_weight_test.tolist()
    x_train = np.append(x_train, [np.fliplr(x) for x in x_test[index_test]],
                        axis=0).astype(np.float32)
    y_train = np.append(y_train, [np.fliplr(x) for x in
                                  np.where(preds_test > 0.5, preds_test,
                                           np.zeros(preds_test.shape))[
                                      index_test]], axis=0).astype(np.float32)
    depth_train = np.append(depth_train, depth_test[index_test], axis=0).astype(np.float32)
    sample_weight_test = np.where(depth_test[index_test, 2] > 0.6, 0.3, 0.05)
    sample_weight = sample_weight + sample_weight_test.tolist()



    modelname2 = 'cache/unet10lov_' + str(model_count) + '.h5'
    modelname3 = 'cache/unet10mse_' + str(model_count) + '.h5'

    model.load_weights(modelname)
    model2.compile(loss=mycustom2, optimizer=RMSprop(lr=0.0005))
    reduce_lr = ReduceLROnPlateau(factor=0.15, patience=8, min_lr=0.00005,
                                  verbose=1)
    model_checkpoint = ModelCheckpoint(modelname2, save_weights_only=True,
                                       save_best_only=True, verbose=1,
                                       mode='max', monitor='iou')
    lh = LossHistory([x_valid2, depth_valid2], y_valid_org, model2,thre=0.0)
    history = model2.fit([x_train, depth_train], y_train,
                         validation_data=[[x_valid2, depth_valid2], y_valid2],
                         epochs=epochs,
                         batch_size=batch_size,
                         sample_weight=np.array(sample_weight),
                         callbacks=[early_stopping, reduce_lr, lh,
                                    model_checkpoint])
    model.load_weights(modelname)
    model.compile(loss=mycustom, optimizer=RMSprop(lr=0.001))
    reduce_lr = ReduceLROnPlateau(factor=0.15, patience=8, min_lr=0.00005,
                                  verbose=1)
    model_checkpoint = ModelCheckpoint(modelname3, save_weights_only=True,
                                       save_best_only=True, verbose=1,
                                       mode='max', monitor='iou')
    lh = LossHistory([x_valid2, depth_valid2], y_valid_org, model)
    history = model.fit([x_train, depth_train], y_train,
                         validation_data=[[x_valid2, depth_valid2], y_valid2],
                         epochs=epochs,
                         batch_size=batch_size,
                         sample_weight=np.array(sample_weight),
                         callbacks=[early_stopping, reduce_lr, lh,
                                    model_checkpoint])


    index_best = (0.5, 0.005263157894736842)
    model2.load_weights(modelname2)
    res = model.predict([x_valid2, depth_valid2], batch_size=40)
    res = np.array([downsample(x) for x in res], dtype=np.float32)
    length = x_valid.shape[0]
    preds_test = (res[:length] + np.array(
        [np.fliplr(x) for x in res[length:2 * length]]) * 0.6 + np.array(
        [downsample2(upsample(x)).reshape((img_size_ori, img_size_ori, 1)) for x in res[length * 2:]]) * 0.3
                  ) / 1.9
    print iou_metric_batch(np.int32(y_valid_org > 0.5), np.int32(preds_test > index_best[0]), index_best[1])
    preds_test.astype(np.float32).tofile("cache/unet10lov_" + str(model_count) + "_valid_np")


    x_test_org = x_test.copy()
    depth_test_org = depth_test.copy()
    x_test2 = np.append(x_test, [np.fliplr(x) for x in x_test_org], axis=0)
    depth_test2 = np.append(depth_test, np.array(map(lambda x: x, depth_test_org)), axis=0).astype(np.float32)
    x_test2 = np.append(x_test2,
                         [upsample2(downsample(x)).reshape((img_size_target, img_size_target, 1)) for x in x_test_org],
                         axis=0).astype(np.float32)
    depth_test2 = np.append(depth_test2, np.array(map(lambda x: x, depth_test_org)), axis=0).astype(np.float32)

    res = model.predict([x_test2, depth_test2], batch_size=40)
    res = np.array([downsample(x) for x in res], dtype=np.float32)
    length = x_test.shape[0]
    preds_test = (res[:length] + np.array(
        [np.fliplr(x) for x in res[length:2 * length]]) * 0.6 + np.array([downsample2(upsample(x)).reshape((img_size_ori, img_size_ori, 1)) for x in res[length * 2:]]) * 0.3
                       )/1.9
    preds_test = np.array([downsample(x) for x in preds_test], dtype=np.float32)
    preds_test.tofile("cache/unet10lov_" + str(model_count) + "_np")
    print iou_metric_batch(np.int32(y_valid_org > 0.5), np.int32(preds_test > index_best[0]), index_best[1])

    index_best = (0.5, 0.005263157894736842)
    model.load_weights(modelname3)
    res = model.predict([x_valid2, depth_valid2], batch_size=40)
    res = np.array([downsample(x) for x in res], dtype=np.float32)
    length = x_valid.shape[0]
    preds_test = (res[:length] + np.array(
        [np.fliplr(x) for x in res[length:2 * length]]) * 0.6 + np.array(
        [downsample2(upsample(x)).reshape((img_size_ori, img_size_ori, 1)) for x in res[length * 2:]]) * 0.3
                  ) / 1.9
    preds_test.astype(np.float32).tofile("cache/unet10mse_" + str(model_count) + "_valid_np")
    print iou_metric_batch(np.int32(y_valid_org > 0.5), np.int32(preds_test > index_best[0]), index_best[1])

    res = model.predict([x_test2, depth_test2], batch_size=40)
    res = np.array([downsample(x) for x in res], dtype=np.float32)
    length = x_test.shape[0]
    preds_test = (res[:length] + np.array(
        [np.fliplr(x) for x in res[length:2 * length]]) * 0.6 + np.array([downsample2(upsample(x)).reshape((img_size_ori, img_size_ori, 1)) for x in res[length * 2:]]) * 0.3
                       )/1.9
    preds_test = np.array([downsample(x) for x in preds_test], dtype=np.float32)
    preds_test.tofile("cache/unet10mse_" + str(model_count) + "_np")
    print iou_metric_batch(np.int32(y_valid_org > 0.5),
                           np.int32(preds_test > index_best[0]), index_best[1])




exit()
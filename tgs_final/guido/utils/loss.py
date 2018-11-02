#!/usr/bin/env python
# -*- coding: utf-8 -*-

from itertools import filterfalse

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# --------------------------- Define Metric ---------------------------


def iou_metric2(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    # true_objects = 2
    # pred_objects = 2

    # bins=([0,0.5,1], [0,0.5, 1])
    # intersection = np.histogram2d(
    #     labels.flatten(), y_pred.flatten(), bins=(true_objects,
    #                                               pred_objects))[0]
    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))[0]

    # Compute areas (needed for finding the union between all objects)
    # area_true = np.histogram(labels, bins=true_objects)[0]
    # area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(
            false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
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


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects,
                                                  pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(
            false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
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


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def iou_metric_batch2(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric2(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def get_iou_vector(preds, labels):
    batch_size = preds.shape[0]
    metric = []
    # ious = []
    for batch in range(batch_size):
        t, p = preds[batch] > 0, labels[batch] > 0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        # iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        iou = (float((intersection > 0).sum()) + 1e-10) / (float(
            (union > 0).sum()) + 1e-10)
        s = []
        for thresh in np.arange(0.5, 1, 0.05):
            s.append(iou > thresh)
        metric.append(np.mean(s))
        # ious.append(iou)

    return np.mean(metric)
    # return np.mean(ious)


def do_kaggle_metric(predict, truth, threshold=0.5, eps=1e-7):
    N = len(predict)
    predict = predict.reshape(N, -1)
    truth = truth.reshape(N, -1)

    predict = predict > threshold
    truth = truth > 0.5
    intersection = truth & predict
    union = truth | predict
    iou = intersection.sum(1) / (union.sum(1) + eps)

    # -------------------------------------------
    result = []
    precision = []
    is_empty_truth = (truth.sum(1) == 0)
    is_empty_predict = (predict.sum(1) == 0)

    # threshold = np.array(
    #     [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    threshold = np.arange(0.5, 1.0, 0.05)

    for t in threshold:
        p = iou >= t

        tp = (~is_empty_truth) & (~is_empty_predict) & (iou > t)
        fp = (~is_empty_truth) & (~is_empty_predict) & (iou <= t)
        fn = (~is_empty_truth) & (is_empty_predict)
        fp_empty = (is_empty_truth) & (~is_empty_predict)
        tn_empty = (is_empty_truth) & (is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append(np.column_stack((tp, fp, fn, tn_empty, fp_empty)))
        precision.append(p)

    result = np.array(result).transpose(1, 2, 0)
    precision = np.column_stack(precision)  # (790, 10)
    precision = precision.mean(1)  # (790, )

    return precision.mean()
    return precision, result, threshold


# --------------------------- BINARY LOSSES ---------------------------

# ---------------
# Lovasz loss
# ---------------


def dscriterion(logit_fuse,
                logit_pixel,
                logit_image,
                truth_pixel,
                truth_image,
                is_average=True):
    loss_fuse = lovasz_loss(
        logit_fuse, truth_pixel, per_image=True, is_average=True)  # TODO
    # loss_fuse = FocalLoss(gamma=2, is_average=True)(logit_fuse, truth_pixel)

    loss_pixel = lovasz_loss(
        logit_pixel, truth_pixel, per_image=True, is_average=False)  # TODO
    # loss_pixel = FocalLoss(gamma=2, is_average=False)(logit_pixel, truth_pixel)

    loss_pixel = loss_pixel * truth_image  # loss for empty image is weighted 0
    if is_average:
        loss_pixel = loss_pixel.sum() / (truth_image.sum() + 1e-6)

    loss_image = F.binary_cross_entropy_with_logits(
        logit_image, truth_image, reduce=is_average)

    weight_fuse, weight_pixel, weight_image = 1, 0.5, 0.05

    # return weight_fuse * loss_fuse, weight_d * loss_d, weight_image * loss_image
    return weight_fuse * loss_fuse, weight_pixel * loss_pixel, weight_image * loss_image


# ---------------
# Lovasz loss
# ---------------
def lovasz_softmax2(probas,
                    labels,
                    only_present=False,
                    per_image=False,
                    ignore=None,
                    is_average=False):
    if per_image:
        if is_average:
            loss = mean(
                lovasz_softmax_flat(
                    *flatten_probas(
                        prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                    only_present=only_present)
                for prob, lab in zip(probas, labels))
        else:
            # loss = torch.cuda.FloatTensor([
            #     lovasz_softmax_flat(
            #         *flatten_probas(
            #             prob.unsqueeze(0), lab.unsqueeze(0), ignore),
            #         only_present=only_present)
            #     for prob, lab in zip(probas, labels)
            # ])
            # print(loss)

            iloss = iter(
                lovasz_softmax_flat(
                    *flatten_probas(
                        prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                    only_present=only_present)
                for prob, lab in zip(probas, labels))
            loss = torch.Tensor(len(probas))
            for i, l in enumerate(iloss):
                loss[i] = l
            loss = loss.cuda()
    else:
        loss = lovasz_softmax_flat(
            *flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax(probas,
                   labels,
                   only_present=False,
                   per_image=False,
                   ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                only_present=only_present)
            for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(
            *flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()  # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(
            torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3,
                            1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# ---------------
# Lovasz loss
# ---------------
def lovasz_loss(logits, labels, per_image=True, ignore=None, is_average=False):
    if per_image:
        if is_average:
            loss = mean(
                lovasz_hinge_flat(*flatten_binary_scores(
                    log.unsqueeze(0), lab.unsqueeze(0), ignore))
                for log, lab in zip(logits, labels))
        else:
            # loss = torch.cuda.FloatTensor([
            #     lovasz_hinge_flat(*flatten_binary_scores(
            #         log.unsqueeze(0), lab.unsqueeze(0), ignore))
            #     for log, lab in zip(logits, labels)
            # ])
            iloss = iter(
                lovasz_hinge_flat(*flatten_binary_scores(
                    log.unsqueeze(0), lab.unsqueeze(0), ignore))
                for log, lab in zip(logits, labels))
            loss = torch.Tensor(len(logits))
            for i, l in enumerate(iloss):
                loss[i] = l
            loss = loss.cuda()
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(*flatten_binary_scores(
                log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


# BCEWithLogitsLoss (has sigmoid)
def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# ---------------
# Mixed dice bce loss
# ---------------
class MixedDiceBceLoss:
    def __init__(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = dice_loss

    def __call__(self, logits, labels):
        bce = self.bce_loss(logits, labels)
        dice = self.dice_loss(logits, labels)

        # eps = 1e-15
        # dice_target = (targets == 1).float()
        # dice_output = outputs
        # intersection = (dice_output * dice_target).sum()
        # union = dice_output.sum() + dice_target.sum() + eps
        # loss -= torch.log(2 * intersection / union)

        loss = 1 + bce - dice

        return loss


def mixed_dice_bce_loss(logits, labels, dice_weight=0.5, bce_weight=0.5):
    bce_loss = nn.BCEWithLogitsLoss()  # has sigmoid
    return dice_weight * dice_loss(logits, labels) + bce_weight * bce_loss(
        logits, labels)
    # return bce_loss(logits, labels) - torch.log(dice_loss(logits, labels))


def dice_loss(logits, labels, smooth=0, eps=1e-7, is_score=False):
    preds = torch.sigmoid(logits)
    # labels.data = labels.data.float()
    intersection = torch.sum(preds * labels)

    score = (2 * intersection + smooth) / (
        torch.sum(preds) + torch.sum(labels) + smooth + eps)
    if is_score:
        return score
    return 1 - score


# ---------------
# Focal loss
# ---------------
# def dice_score(input, target):
#     input = torch.sigmoid(input)
#     smooth = 1.0

#     iflat = input.view(-1)
#     tflat = target.view(-1)
#     intersection = (iflat * tflat).sum()

#     return ((2.0 * intersection + smooth) /
#             (iflat.sum() + tflat.sum() + smooth))


# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
# Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
class FocalLoss__(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss__, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            # BCE_loss = F.binary_cross_entropy_with_logits(
            #     inputs, targets, reduction="none")
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduce=False)
        else:
            # BCE_loss = F.binary_cross_entropy(
            #     inputs, targets, reduction="none")
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma, is_average=True):
        super().__init__()
        self.gamma = gamma
        self.is_average = is_average

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()))

        # input = torch.sigmoid(input)

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        if self.is_average:
            return loss.mean()
        else:
            loss = loss.view(len(input), -1)
            loss = loss.mean(1)
            return loss


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma, logits=True)
        # self.focal_ = FocalLoss_(gamma)

    def forward(self, input, target):
        # loss = self.alpha * self.focal(input, target) - torch.log(
        #     dice_score(input, target))  # dice_loss score
        # return loss.mean()
        loss = self.focal(input, target)
        return loss


# https://xmfbit.github.io/2017/08/14/focal-loss-paper/
# https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
class FocalLossasdf(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)
            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))
        if target.dim() > 2:
            target = target.view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2))

        size = (target.size(0), input.size(-1) + 1)
        view = (target.size(0), 1)
        mask = input.data.new(*size).fill_(0)
        target = target.view(*view).long()  # long
        ones = 1.
        y = mask.scatter_(1, target, ones)
        y = y[:, 1:]  #

        logit = torch.sigmoid(input)  # F.softmax(input)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit)**self.gamma  # focal loss
        return loss.mean()


class FocalLoss_now(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, eps=1e-7,
                 size_average=True):  # alpha=0.25
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.size_average = size_average
        print("FOCAL LOSS", gamma, alpha)

    def forward(self, input, target):
        if input.dim() == 1:
            input = input.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)
            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))
        if target.dim() > 2:
            target = target.view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2))

        P = torch.sigmoid(input)
        P = P.clamp(self.eps, 1. - self.eps)

        pt = P * target + (1 - P) * (1 - target)
        # pt = torch.clamp(pt, 0.0001, 0.9999)
        logpt = pt.log()
        at = (1 - self.alpha) * target + (self.alpha) * (1 - target)
        logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# --------------------------- HELPER FUNCTIONS ---------------------------


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds, ), (labels, )
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        # union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        union = ((label == 1) | (pred == 1)).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    # return 100 * iou
    return iou


def mean(il, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    il = iter(il)
    if ignore_nan:
        il = filterfalse(np.isnan, il)
    try:
        n = 1
        acc = next(il)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(il, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

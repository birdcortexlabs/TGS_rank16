#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
# import copy
import math
import shutil

from bisect import bisect_right
import numpy as np

from tensorboardX import SummaryWriter

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils import data
from torch.autograd import Variable
# import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold

from utils import unet_models
from utils.dataset import TGSSaltDataset
from utils.im_utils import transform
from utils.loss import lovasz_hinge, FocalLoss, get_iou_vector, dscriterion
from utils.tools import cuda, to_item, histcoverage, save_state, load_state


def train(model,
          train_dataloader,
          epoch,
          num_epochs,
          criterion,
          optimizer,
          is_finetune,
          scheduler=None,
          writer=None):
    model.train()

    # # do
    # def set_bn_eval(model):
    #     for name, module in model.named_children():
    #         if isinstance(module, torch.nn.BatchNorm2d):
    #             module.eval()
    #             module.weight.requires_grad = False
    #             module.bias.requires_grad = False
    #         else:
    #             set_bn_eval(module)

    # # do
    # for name, module in model.named_children():
    #     if "encoder" in name:
    #         set_bn_eval(module)

    train_loss = []
    train_iout = []
    for batch_idx, (inputs, labels, nemptys,
                    indexs) in enumerate(train_dataloader):
        if scheduler is not None:
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            step = (epoch - 1) * len(train_dataloader) + batch_idx
            writer.add_scalar('lr/lr', lr, step)

        inputs = cuda(inputs)
        with torch.no_grad():
            labels = cuda(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        logits = model(inputs)
        preds = torch.sigmoid(logits)

        loss = criterion(logits, labels)

        if not is_finetune:
            iout = get_iou_vector((preds > 0.5), labels)
        else:
            iout = get_iou_vector((logits > 0), labels)

        loss.backward()
        optimizer.step()

        train_loss.append(to_item(loss))
        train_iout.append(to_item(iout))

        sys.stdout.write('\r')
        sys.stdout.write(
            f'| Epoch [{epoch:3d}/{num_epochs:3d}] '
            f'Iter[{(batch_idx + 1):3d}/{len(train_dataloader):3d}]\t\t'
            f'Loss: {np.mean(train_loss):.4f} '
            f'IoUT: {np.mean(train_iout):.4f}')
        sys.stdout.flush()
    print()
    metrics = {
        'train_loss': np.mean(train_loss),
        'train_iout': np.mean(train_iout)
    }
    return metrics


def evaluate(model, valid_dataloader, epoch, criterion, is_finetune):
    with torch.no_grad():
        model.eval()
        valid_loss = []
        valid_iout = []
        for inputs, labels, nemptys, indexs in valid_dataloader:
            inputs = cuda(inputs)
            labels = cuda(labels)

            logits = model(inputs)
            preds = torch.sigmoid(logits)

            loss = criterion(logits, labels)

            if not is_finetune:
                iout = get_iou_vector((preds > 0.5), labels)
            else:
                iout = get_iou_vector((logits > 0), labels)

            valid_loss.append(to_item(loss))
            valid_iout.append(to_item(iout))

        print(f'| Validation Epoch #{epoch}\t\t\t'
              f'Valid Loss: {np.mean(valid_loss):.4f} '
              f'Valid IoUT: {np.mean(valid_iout):.4f}')

        metrics = {
            'valid_loss': np.mean(valid_loss),
            'valid_iout': np.mean(valid_iout)
        }
        return metrics


def train2(model, train_dataloader, epoch, num_epochs, optimizer):
    model.train()

    train_loss = []
    train_iout = []
    train_iout2 = []
    train_iout3 = []
    for batch_idx, (inputs, labels, nemptys,
                    indexs) in enumerate(train_dataloader):

        inputs = cuda(inputs)
        with torch.no_grad():
            nemptys = Variable(nemptys).type(torch.cuda.FloatTensor)
            labels = cuda(labels)

        optimizer.zero_grad()

        logit_fuse, logit_pixel, logit_image = model(inputs)
        preds_fuse = torch.sigmoid(logit_fuse)
        preds_image = torch.sigmoid(logit_image)

        truth_pixel, truth_image = labels, nemptys
        loss_fuse, loss_pixel, loss_image = dscriterion(
            logit_fuse, logit_pixel, logit_image, truth_pixel, truth_image)

        preds_image = preds_image.view(-1, 1, 1, 1)
        # preds_pixel = torch.mul(logit_pixel, (preds_image > 0.5).float())
        preds_fuse2 = torch.mul(preds_fuse, (preds_image > 0.5).float())

        iout = get_iou_vector((logit_fuse > 0), labels)  # TODO
        # iout2 = get_iou_vector((preds_pixel > 0), labels)
        iout2 = get_iou_vector((preds_fuse > 0.5), labels)
        iout3 = get_iou_vector((preds_fuse2 > 0.5), labels)

        loss = loss_fuse + loss_pixel + loss_image
        loss.backward()
        # loss_pixel.backward(retain_graph=True)
        # loss_image.backward()
        optimizer.step()

        train_loss.append(to_item(loss))
        train_iout.append(to_item(iout))
        train_iout2.append(to_item(iout2))
        train_iout3.append(to_item(iout3))

        sys.stdout.write('\r')
        sys.stdout.write(
            f'| Epoch [{epoch:3d}/{num_epochs:3d}] '
            f'Iter[{(batch_idx + 1):3d}/{len(train_dataloader):3d}]\t\t'
            f'Loss: {np.mean(train_loss):.4f} '
            f'{np.mean(train_iout):.4f} '
            f'{np.mean(train_iout2):.4f} '
            f'{np.mean(train_iout3):.4f}')
        sys.stdout.flush()
    print()
    metrics = {
        'train_loss': np.mean(train_loss),
        'train_iout': np.mean(train_iout)
    }
    return metrics


def evaluate2(model, valid_dataloader, epoch):
    with torch.no_grad():
        model.eval()
        valid_loss = []
        valid_iout = []
        valid_iout2 = []
        valid_iout3 = []
        for inputs, labels, nemptys, indexs in valid_dataloader:
            inputs = cuda(inputs)
            nemptys = Variable(nemptys).type(torch.cuda.FloatTensor)
            labels = cuda(labels)

            logit_fuse, logit_pixel, logit_image = model(inputs)
            preds_fuse = torch.sigmoid(logit_fuse)
            preds_image = torch.sigmoid(logit_image)

            truth_pixel, truth_image = labels, nemptys
            loss_fuse, loss_pixel, loss_image = dscriterion(
                logit_fuse, logit_pixel, logit_image, truth_pixel, truth_image)

            preds_image = preds_image.view(-1, 1, 1, 1)
            # preds_pixel = torch.mul(logit_pixel, (preds_image > 0.5).float())
            preds_fuse2 = torch.mul(preds_fuse, (preds_image > 0.5).float())

            iout = get_iou_vector((logit_fuse > 0), labels)
            # iout2 = get_iou_vector((preds_pixel > 0), labels)
            iout2 = get_iou_vector((preds_fuse > 0.5), labels)
            iout3 = get_iou_vector((preds_fuse2 > 0.5), labels)

            loss = loss_fuse + loss_pixel + loss_image

            valid_loss.append(to_item(loss))
            valid_iout.append(to_item(iout))
            valid_iout2.append(to_item(iout2))
            valid_iout3.append(to_item(iout3))

        print(f'| Validation Epoch #{epoch}\t\t\t'
              f'Valid Loss: {np.mean(valid_loss):.4f} '
              f'Valid IoUT: {np.mean(valid_iout):.4f} '
              f'Valid IoUT2: {np.mean(valid_iout2):.4f} '
              f'Valid IoUT3: {np.mean(valid_iout3):.4f}')

        metrics = {
            'valid_loss': np.mean(valid_loss),
            'valid_iout': np.mean(valid_iout)
        }
        return metrics


def adjust_lr(optimizer,
              epoch,
              init_lr=1e-4,
              num_epochs_per_decay=20,
              lr_decay_factor=0.5):  # num_epochs_per_decay=12
    lr = init_lr * (lr_decay_factor**(epoch // num_epochs_per_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def cyclic_lr(
        optimizer,
        epoch,
        init_lr=1e-4,
        num_epochs_per_cycle=6,  # 5
        cycle_epochs_decay=2,
        lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor**(epoch_in_cycle // cycle_epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# https://github.com/Harshvardhan1/cyclic-learning-schedulers-pytorch
class CyclicCosAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, milestones, eta_min=0, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.eta_min = eta_min
        self.milestones = milestones
        super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones, self.last_epoch)

        left_barrier = 0 if idx == 0 else self.milestones[idx - 1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch - left_barrier

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * curr_pos / width)) / 2
            for base_lr in self.base_lrs
        ]


def train_model(model, train_dataloader, valid_dataloader, fold=None):
    num_epochs = 30

    print("| -- Training Model -- |")

    model_path = os.path.join("model", f"model_{fold}.pt")
    best_model_path = os.path.join("model", f"best_model_{fold}.pt")

    model, start_epoch, max_iout = load_state(model, best_model_path)

    writer = SummaryWriter()

    # define loss function (criterion) and optimizer
    # criterion = FocalLoss(gamma=2, logits=True)
    criterion = FocalLoss(gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # best_model_wts = copy.deepcopy(model.state_dict())  # deepcopy important

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print(f"| Epoch [{epoch:3d}/{num_epochs:3d}], lr={lr}")

        train_metrics = train(model, train_dataloader, epoch, num_epochs,
                              criterion, optimizer, False)
        valid_metrics = evaluate(model, valid_dataloader, epoch, criterion,
                                 False)

        writer.add_scalar('loss/train', train_metrics["train_loss"], epoch)
        writer.add_scalar('iout/train', train_metrics["train_iout"], epoch)
        writer.add_scalar('loss/valid', valid_metrics["valid_loss"], epoch)
        writer.add_scalar('iout/valid', valid_metrics["valid_iout"], epoch)

        valid_iout = valid_metrics["valid_iout"]

        save_state(model, epoch, valid_iout, model_path)

        if valid_iout > max_iout:
            max_iout = valid_iout
            # best_model_wts = copy.deepcopy(model.state_dict())
            shutil.copyfile(model_path, best_model_path)
            print(f"|- Save model, Epoch #{epoch}, max_iout: {max_iout:.4f}")

        # if valid_iout > 0.8:
        #     break


def finetune(model, train_dataloader, valid_dataloader, fold=None):
    num_epochs = 150

    print("| -- Finetune Model -- |")

    best_model_path = os.path.join("model", f"best_model_{fold}.pt")

    model, _, max_iout = load_state(model, best_model_path)
    start_epoch = 1

    model_path = os.path.join("model", f"model_{fold}_finetune.pt")
    best_model_path = os.path.join("model", f"best_model_{fold}_finetune.pt")

    writer = SummaryWriter()

    # define loss function (criterion) and optimizer
    criterion = lovasz_hinge
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-5)

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print(f"| Epoch [{epoch:3d}/{num_epochs:3d}], lr={lr}")

        train_metrics = train(model, train_dataloader, epoch, num_epochs,
                              criterion, optimizer, True)
        valid_metrics = evaluate(model, valid_dataloader, epoch, criterion,
                                 True)

        writer.add_scalar('loss/train2', train_metrics["train_loss"], epoch)
        writer.add_scalar('iout/train2', train_metrics["train_iout"], epoch)
        writer.add_scalar('loss/valid2', valid_metrics["valid_loss"], epoch)
        writer.add_scalar('iout/valid2', valid_metrics["valid_iout"], epoch)

        valid_iout = valid_metrics["valid_iout"]

        scheduler.step(valid_iout)

        save_state(model, epoch, valid_iout, model_path)

        if valid_iout > max_iout:
            max_iout = valid_iout
            shutil.copyfile(model_path, best_model_path)
            print(f"|- Save model, Epoch #{epoch}, max_iout: {max_iout:.4f}")


def finetune2(model, train_dataloader, valid_dataloader, fold=None):
    cycles = 6  # 6
    per_cycle = 50  # 50
    num_epochs = cycles * per_cycle  # 300
    mini_batches = len(train_dataloader)

    print("| -- Finetune2 Model -- |")

    best_model_path = os.path.join("model", f"best_model_{fold}_finetune.pt")

    model, _, _ = load_state(model, best_model_path)
    start_epoch = 1

    model_path = os.path.join("model", f"model_{fold}_finetune2.pt")

    writer = SummaryWriter()

    # define loss function (criterion) and optimizer
    criterion = lovasz_hinge
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    milestones = [(i + 1) * per_cycle * mini_batches for i in range(cycles)]
    scheduler = CyclicCosAnnealingLR(optimizer, milestones, eta_min=1e-5)
    print(milestones)

    max_iout = 0.84

    for epoch in range(start_epoch, num_epochs + 1):
        train_metrics = train(model, train_dataloader, epoch, num_epochs,
                              criterion, optimizer, True, scheduler, writer)
        valid_metrics = evaluate(model, valid_dataloader, epoch, criterion,
                                 True)

        writer.add_scalar('loss/train3', train_metrics["train_loss"], epoch)
        writer.add_scalar('iout/train3', train_metrics["train_iout"], epoch)
        writer.add_scalar('loss/valid3', valid_metrics["valid_loss"], epoch)
        writer.add_scalar('iout/valid3', valid_metrics["valid_iout"], epoch)

        valid_iout = valid_metrics["valid_iout"]

        save_state(model, epoch, valid_iout, model_path)

        if valid_iout > max_iout:
            max_iout = valid_iout
            best_model_path = os.path.join(
                "model",
                f"best_model_{fold}_finetune2_{(epoch-1)//per_cycle}.pt")
            shutil.copyfile(model_path, best_model_path)
            print(f"|- Save model, Epoch #{epoch}, max_iout: {max_iout:.4f}")

        if epoch % per_cycle == 0:
            max_iout = 0.84

    return cycles


def train_model2(model, train_dataloader, valid_dataloader, fold=None):
    num_epochs = 150

    print("| -- Training Model -- |")

    model_path = os.path.join("model", f"model_{fold}.pt")
    best_model_path = os.path.join("model", f"best_model_{fold}.pt")

    model, start_epoch, max_iout = load_state(model, best_model_path)

    writer = SummaryWriter()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-5)

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print(f"| Epoch [{epoch:3d}/{num_epochs:3d}], lr={lr}")

        train_metrics = train2(model, train_dataloader, epoch, num_epochs,
                               optimizer)
        valid_metrics = evaluate2(model, valid_dataloader, epoch)

        writer.add_scalar('loss/train', train_metrics["train_loss"], epoch)
        writer.add_scalar('iout/train', train_metrics["train_iout"], epoch)
        writer.add_scalar('loss/valid', valid_metrics["valid_loss"], epoch)
        writer.add_scalar('iout/valid', valid_metrics["valid_iout"], epoch)

        valid_iout = valid_metrics["valid_iout"]

        scheduler.step(valid_iout)

        save_state(model, epoch, valid_iout, model_path)

        if valid_iout > max_iout:
            max_iout = valid_iout
            shutil.copyfile(model_path, best_model_path)
            print(f"|- Save model, Epoch #{epoch}, max_iout: {max_iout:.4f}")


def train_fold(train_df, ids_trains, train_dir, batch_size, n_fold, phase):
    # Create train/validation split stratified by salt coverage
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=1234)

    for fold, (train_idx, valid_idx) in enumerate(
            skf.split(ids_trains, train_df.coverage_class)):
        ids_train, ids_valid = ids_trains[train_idx], ids_trains[valid_idx]

        if n_fold != -1 and fold != n_fold:
            continue

        histall = histcoverage(train_df.coverage_class[ids_train].values)
        histall_valid = histcoverage(train_df.coverage_class[ids_valid].values)
        print(f"fold: {fold}\n"
              f"train size: {len(ids_train)}\n"
              f"number of each mask class: {histall}\n"
              f"valid size: {len(ids_valid)}\n"
              f"number of each mask class: {histall_valid}")

        train_dataset = TGSSaltDataset(
            train_dir, ids_train, transform, mode='train')
        valid_dataset = TGSSaltDataset(
            train_dir, ids_valid, None, mode='train')

        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True)
        valid_dataloader = data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True)
        # imshow_example(train_dataloader)
        # exit()

        model = unet_models.DeepSupervision50(pretrained=True)
        print(sum(p.numel() for p in model.parameters()))
        model = cuda(model)

        if phase == "train2":
            train_model2(model, train_dataloader, valid_dataloader, fold=fold)
        elif phase == "train":
            train_model(model, train_dataloader, valid_dataloader, fold=fold)
        elif phase == "finetune":
            finetune(model, train_dataloader, valid_dataloader, fold=fold)
        elif phase == "finetune2":
            finetune2(model, train_dataloader, valid_dataloader, fold=fold)


# cycle_start_epoch = 60

# if epoch == cycle_start_epoch:
#     optimizer = optim.Adam(model.parameters(), lr=0.000025)
#     # optimizer = optim.Adam(model.parameters(), lr=0.00005)
# if epoch >= cycle_start_epoch:
#     lr = cyclic_lr(
#         optimizer, epoch - cycle_start_epoch, init_lr=0.000025)
#     # optimizer, epoch - cycle_start_epoch, init_lr=0.00005)
# else:
#     lr = adjust_lr(optimizer, epoch)

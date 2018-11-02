#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from utils.loss import lovasz_loss
from utils.unet_utils import ConvRelu, ConvBn2d, UnetConv2, UnetUp
from utils.unet_utils import DecoderBlock, DecoderBlockLinkNet, DecoderHyper
from utils.unet_utils import UnetGridGatingSignal, MultiAttentionBlock

import pretrainedmodels

# UNet11, UNet16
# AlbuNet34, UNetResNet(34, 101, 152, pretrained=True, is_deconv=True)
# LinkNet34(pretrained=True), NewLinkNet34(pretrained=True)
# LinkNetAttention(pretrained=True), UNetAttention()
# HyperResnet34, HyperResnet34_v2
# HyperResnext50(pretrained=True)


class UNet11(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 pretrained=False,
                 is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG11
        """
        super().__init__()

        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.encoder[0],
            self.relu,
        )
        self.conv2 = nn.Sequential(
            self.encoder[3],
            self.relu,
        )
        self.conv3 = nn.Sequential(
            self.encoder[6],
            self.relu,
            self.encoder[8],
            self.relu,
        )
        self.conv4 = nn.Sequential(
            self.encoder[11],
            self.relu,
            self.encoder[13],
            self.relu,
        )
        self.conv5 = nn.Sequential(
            self.encoder[16],
            self.relu,
            self.encoder[18],
            self.relu,
        )

        self.center = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2,
                                   num_filters * 8, is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2,
                                 num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2,
                                 num_filters * 4, is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 4, num_filters * 4 * 2,
                                 num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2,
                                 num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out


class UNet16(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 pretrained=False,
                 is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()

        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.encoder[0],
            self.relu,
            self.encoder[2],
            self.relu,
        )
        self.conv2 = nn.Sequential(
            self.encoder[5],
            self.relu,
            self.encoder[7],
            self.relu,
        )
        self.conv3 = nn.Sequential(
            self.encoder[10],
            self.relu,
            self.encoder[12],
            self.relu,
            self.encoder[14],
            self.relu,
        )
        self.conv4 = nn.Sequential(
            self.encoder[17],
            self.relu,
            self.encoder[19],
            self.relu,
            self.encoder[21],
            self.relu,
        )
        self.conv5 = nn.Sequential(
            self.encoder[24],
            self.relu,
            self.encoder[26],
            self.relu,
            self.encoder[28],
            self.relu,
        )

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2,
                                 num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2,
                                 num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(256 + num_filters * 8, num_filters * 4 * 2,
                                 num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(128 + num_filters * 2, num_filters * 2 * 2,
                                 num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out


class AlbuNet34(nn.Module):
    """
    UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    """

    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 pretrained=False,
                 is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = models.resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1,
                                   self.encoder.relu, self.pool)  # pool ?
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(512, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)

        self.dec5 = DecoderBlock(512 + num_filters * 8, num_filters * 8 * 2,
                                 num_filters * 8, is_deconv)
        self.dec4 = DecoderBlock(256 + num_filters * 8, num_filters * 8 * 2,
                                 num_filters * 8, is_deconv)
        self.dec3 = DecoderBlock(128 + num_filters * 8, num_filters * 4 * 2,
                                 num_filters * 2, is_deconv)
        self.dec2 = DecoderBlock(64 + num_filters * 2, num_filters * 2 * 2,
                                 num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2,
                                 num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out


class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.
    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.
    """

    def __init__(self,
                 encoder_depth,
                 num_classes=1,
                 num_filters=32,
                 dropout_2d=0.2,
                 pretrained=False,
                 is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError(
                'only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1,
                                   self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlock(bottom_channel_nr, num_filters * 8 * 2,
                                   num_filters * 8, is_deconv)
        self.dec5 = DecoderBlock(bottom_channel_nr + num_filters * 8,
                                 num_filters * 8 * 2, num_filters * 8,
                                 is_deconv)
        self.dec4 = DecoderBlock(bottom_channel_nr // 2 + num_filters * 8,
                                 num_filters * 8 * 2, num_filters * 8,
                                 is_deconv)
        self.dec3 = DecoderBlock(bottom_channel_nr // 4 + num_filters * 8,
                                 num_filters * 4 * 2, num_filters * 2,
                                 is_deconv)
        self.dec2 = DecoderBlock(bottom_channel_nr // 8 + num_filters * 2,
                                 num_filters * 2 * 2, num_filters * 2 * 2,
                                 is_deconv)
        self.dec1 = DecoderBlock(num_filters * 2 * 2, num_filters * 2 * 2,
                                 num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)  # 8, 64, 56, 56
        conv2 = self.conv2(conv1)  # 8, 256, 56, 56
        conv3 = self.conv3(conv2)  # 8, 512, 28, 28
        conv4 = self.conv4(conv3)  # 8, 1024, 14, 14
        conv5 = self.conv5(conv4)  # 8, 2048, 7, 7

        center = self.center(self.pool(conv5))  # 8, 256, 6, 6

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        return self.final(F.dropout2d(dec0, p=self.dropout_2d))


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                      resnet.maxpool)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)
        f5 = self.finalconv3(F.dropout2d(f4, p=0.2))

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out


class NewLinkNet34(nn.Module):
    def __init__(self,
                 num_classes=1,
                 num_filters=32,
                 pretrained=True,
                 is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        self.relu = nn.ReLU(inplace=True)  # do
        resnet = models.resnet34(pretrained=pretrained)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])
        self.decoder0 = DecoderBlockLinkNet(filters[0], filters[0] // 2)  # do

        # Final Classifier
        self.final1 = ConvRelu(num_filters, num_filters)
        self.final2 = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    # noinspection PyCallingNonCallable
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        e0 = self.firstrelu(x)
        pool = self.firstmaxpool(e0)
        e1 = self.encoder1(pool)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder with Skip Connections # BasicBlock(+ first, than relu)
        d4 = self.relu(self.decoder4(e4) + e3)  # 256 + relu
        d3 = self.relu(self.decoder3(d4) + e2)  # 128 + relu
        d2 = self.relu(self.decoder2(d3) + e1)  # 64 + relu
        d1 = self.relu(self.decoder1(d2) + e0)  # 64 do
        d0 = self.decoder0(d1)  # 32 do

        # Final Classification
        f1 = self.final1(d0)
        f2 = self.final2(f1)

        if self.num_classes > 1:
            x_out = F.log_softmax(f2, dim=1)
        else:
            x_out = f2
        return x_out


class HyperResnext50_v2(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(HyperResnext50_v2, self).__init__()

        model_name = 'se_resnext50_32x4d'
        se_resnext = pretrainedmodels.__dict__[model_name](
            num_classes=1000, pretrained='imagenet')

        self.encoder0 = se_resnext.layer0
        removed = list(self.encoder0.children())[:-1]  # remove maxpool
        self.encoder0 = nn.Sequential(*removed)
        self.encoder1 = se_resnext.layer1
        self.encoder2 = se_resnext.layer2
        self.encoder3 = se_resnext.layer3
        self.encoder4 = se_resnext.layer4

        self.center = nn.Sequential(
            ConvBn2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = DecoderHyper(256 + 2048, 512, 128)  # 3, 64
        self.decoder4 = DecoderHyper(128 + 1024, 512, 128)
        self.decoder3 = DecoderHyper(128 + 512, 256, 128)
        self.decoder2 = DecoderHyper(128 + 256, 128, 128)
        self.decoder1 = DecoderHyper(128, 64, 128)  # 2, 64?

        self.logit = nn.Sequential(
            nn.Conv2d(640, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        e0 = self.encoder0(x)  # 8, 64, 112, 112
        e1 = self.encoder1(e0)  # 8, 256, 112, 112
        e2 = self.encoder2(e1)  # 8, 512, 56, 56
        e3 = self.encoder3(e2)  # 8, 1024, 28, 28
        e4 = self.encoder4(e3)  # 8, 2048, 14, 14

        f = self.center(e4)  # 8, 256, 7, 7

        d5 = self.decoder5(f, e4)
        d4 = self.decoder4(d5, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)

        f = torch.cat((
            d1,
            F.upsample(
                d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(
                d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(
                d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(
                d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)  # hyper column

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)

        return logit


class HyperResnext50(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(HyperResnext50, self).__init__()

        model_name = 'se_resnext50_32x4d'
        se_resnext = pretrainedmodels.__dict__[model_name](
            num_classes=1000, pretrained='imagenet')

        self.encoder0 = se_resnext.layer0
        removed = list(self.encoder0.children())[:-1]  # remove maxpool
        self.encoder0 = nn.Sequential(*removed)
        self.encoder1 = se_resnext.layer1
        self.encoder2 = se_resnext.layer2
        self.encoder3 = se_resnext.layer3
        self.encoder4 = se_resnext.layer4

        self.center = nn.Sequential(
            ConvBn2d(2048, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = DecoderHyper(256 + 2048, 512, 64)  # 3, 128?
        self.decoder4 = DecoderHyper(64 + 1024, 512, 64)
        self.decoder3 = DecoderHyper(64 + 512, 256, 64)
        self.decoder2 = DecoderHyper(64 + 256, 128, 64)
        self.decoder1 = DecoderHyper(64, 64, 64)  # 2, 64?

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        e0 = self.encoder0(x)  # 8, 64, 112, 112
        e1 = self.encoder1(e0)  # 8, 256, 112, 112
        e2 = self.encoder2(e1)  # 8, 512, 56, 56
        e3 = self.encoder3(e2)  # 8, 1024, 28, 28
        e4 = self.encoder4(e3)  # 8, 2048, 14, 14

        f = self.center(e4)  # 8, 256, 7, 7

        # (8, 256, 7*2, 7*2) + (8, 2048, 14, 14) -> (8, 64, 14, 14)
        d5 = self.decoder5(f, e4)
        # (8, 64, 14*2, 14*2) + (8, 1024, 28, 28) -> (8, 64, 28, 28)
        d4 = self.decoder4(d5, e3)
        # (8, 64, 28*2, 28*2) + (8, 512, 56, 56) -> (8, 64, 56, 56)
        d3 = self.decoder3(d4, e2)
        # (8, 64, 56*2, 56*2) + (8, 256, 112, 112) -> (8, 64, 112, 112)
        d2 = self.decoder2(d3, e1)
        # (8, 64, 112*2, 112*2) -> (8, 64, 224, 224)
        d1 = self.decoder1(d2)

        f = torch.cat((
            d1,
            F.upsample(
                d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(
                d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(
                d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(
                d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)  # hyper column

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)

        return logit


class HyperResnet34_v2(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(HyperResnet34_v2, self).__init__()

        resnet = models.resnet34(pretrained=pretrained)
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # do
        )
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = DecoderHyper(256 + 512, 512, 64)
        self.decoder4 = DecoderHyper(64 + 256, 256, 64)
        self.decoder3 = DecoderHyper(64 + 128, 128, 64)
        self.decoder2 = DecoderHyper(64 + 64, 64, 64)
        self.decoder1 = DecoderHyper(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        f = self.center(e4)

        d5 = self.decoder5(f, e4)
        d4 = self.decoder4(d5, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)

        f = torch.cat((
            F.upsample(
                d1, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(
                d2, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(
                d3, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(
                d4, scale_factor=16, mode='bilinear', align_corners=False),
            F.upsample(
                d5, scale_factor=32, mode='bilinear', align_corners=False),
        ), 1)  # hyper column do

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)

        return logit


class HyperResnet34(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(HyperResnet34, self).__init__()

        resnet = models.resnet34(pretrained=pretrained)
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # resnet.maxpool # remove
        )
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = DecoderHyper(256 + 512, 512, 64)
        self.decoder4 = DecoderHyper(64 + 256, 256, 64)
        self.decoder3 = DecoderHyper(64 + 128, 128, 64)
        self.decoder2 = DecoderHyper(64 + 64, 64, 64)
        self.decoder1 = DecoderHyper(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        e0 = self.encoder0(x)  # 16, 64, 64, 64
        e1 = self.encoder1(e0)  # 16, 64, 64, 64
        e2 = self.encoder2(e1)  # 16, 128, 32, 32
        e3 = self.encoder3(e2)  # 16, 256, 16, 16
        e4 = self.encoder4(e3)  # 16, 512, 8, 8

        f = self.center(e4)  # 16, 256, 4, 4

        d5 = self.decoder5(f, e4)
        d4 = self.decoder4(d5, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2)

        f = torch.cat((
            d1,
            F.upsample(
                d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(
                d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(
                d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(
                d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)  # hyper column

        f = F.dropout2d(f, p=0.50)
        logit = self.logit(f)

        return logit


class DeepSupervision50(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()
        model_name = 'se_resnext50_32x4d'
        se_resnext = pretrainedmodels.__dict__[model_name](
            num_classes=1000, pretrained='imagenet')

        self.encoder0 = se_resnext.layer0
        removed = list(self.encoder0.children())[:-1]  # remove maxpool
        self.encoder0 = nn.Sequential(*removed)
        self.encoder1 = se_resnext.layer1
        self.encoder2 = se_resnext.layer2
        self.encoder3 = se_resnext.layer3
        self.encoder4 = se_resnext.layer4

        self.center = nn.Sequential(
            ConvBn2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = DecoderHyper(256 + 2048, 512, 128)
        self.decoder4 = DecoderHyper(128 + 1024, 512, 128)
        self.decoder3 = DecoderHyper(128 + 512, 256, 128)
        self.decoder2 = DecoderHyper(128 + 256, 128, 128)
        self.decoder1 = DecoderHyper(128, 64, 128)

        self.fuse_pixel = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
        )

        self.logit_pixel = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

        self.fuse_image = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

        self.logit_image = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.logit = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape

        e0 = self.encoder0(x)  # 16, 64, 64, 64
        e1 = self.encoder1(e0)  # 16, 256, 64, 64
        e2 = self.encoder2(e1)  # 16, 512, 32, 32
        e3 = self.encoder3(e2)  # 16, 1024, 16, 16
        e4 = self.encoder4(e3)  # 16, 2048, 8, 8

        f = self.center(e4)  # 16, 256, 4, 4

        d5 = self.decoder5(f, e4)  # 16, 128, 8, 8
        d4 = self.decoder4(d5, e3)  # 16, 128, 16, 16
        d3 = self.decoder3(d4, e2)  # 16, 128, 32, 32
        d2 = self.decoder2(d3, e1)  # 16, 128, 64, 64
        d1 = self.decoder1(d2)  # 16, 128, 128, 128

        # logit_d = list()
        # logit_d.append(d1)
        # logit_d.append(d2)
        # logit_d.append(d3)
        # logit_d.append(d4)
        # logit_d.append(d5)

        d2 = F.upsample(
            d2, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = F.upsample(
            d3, scale_factor=4, mode='bilinear', align_corners=False)
        d4 = F.upsample(
            d4, scale_factor=8, mode='bilinear', align_corners=False)
        d5 = F.upsample(
            d5, scale_factor=16, mode='bilinear', align_corners=False)
        d = torch.cat((d1, d2, d3, d4, d5), 1)  # hyper-columns
        d = F.dropout(d, p=0.50, training=self.training)
        fuse_pixel = self.fuse_pixel(d)  # 16, 128, 128, 128
        logit_pixel = self.logit_pixel(fuse_pixel)  # 16, 1, 128, 128

        e = F.adaptive_avg_pool2d(e4, output_size=1).view(batch_size, -1)
        e = F.dropout(e, p=0.50, training=self.training)  # 16, 2048
        fuse_image = self.fuse_image(e)  # 16, 128
        logit_image = self.logit_image(fuse_image).view(-1)  # 16

        fuse = self.fuse(
            torch.cat([
                fuse_pixel,
                F.upsample(
                    fuse_image.view(batch_size, -1, 1, 1),
                    scale_factor=128,
                    mode='nearest')
            ], 1))  # 16, 64, 128, 128
        # logit = self.logit(fuse)  # 16, 1, 128, 128

        # return logit, logit_pixel, logit_image
        return fuse, logit_pixel, logit_image


class DeepSupervision34(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        resnet = models.resnet34(pretrained=pretrained)
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            # nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
        )
        # self.encoder1 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     resnet.layer1,
        # )
        self.encoder1 = resnet.layer1  #
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.decoder5 = DecoderHyper(256 + 512, 512, 64)
        self.decoder4 = DecoderHyper(64 + 256, 256, 64)
        self.decoder3 = DecoderHyper(64 + 128, 128, 64)
        self.decoder2 = DecoderHyper(64 + 64, 64, 64)
        self.decoder1 = DecoderHyper(64, 32, 64)  # 64 + 64

        self.fuse_pixel = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
        )

        self.logit_pixel = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

        self.fuse_image = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
        )

        self.logit_image = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
        )

        self.logit = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size, C, H, W = x.shape

        e0 = self.encoder0(x)  # 16, 64, 128, 128
        e1 = self.encoder1(e0)  # 16, 64, 64, 64
        e2 = self.encoder2(e1)  # 16, 128, 32, 32
        e3 = self.encoder3(e2)  # 16, 256, 16, 16
        e4 = self.encoder4(e3)  # 16, 512, 8, 8

        f = self.center(e4)  # 16, 256, 4, 4

        d5 = self.decoder5(f, e4)  # 16, 64, 8, 8
        d4 = self.decoder4(d5, e3)  # 16, 64, 16, 16
        d3 = self.decoder3(d4, e2)  # 16, 64, 32, 32
        d2 = self.decoder2(d3, e1)  # 16, 64, 64, 64
        # d1 = self.decoder1(d2, e0)  # 16, 64, 128, 128
        d1 = self.decoder1(d2)  # 16, 64, 128, 128

        d = torch.cat((
            d1,
            F.upsample(
                d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(
                d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(
                d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(
                d5, scale_factor=16, mode='bilinear', align_corners=False),
        ), 1)
        d = F.dropout(d, p=0.50, training=self.training)
        fuse_pixel = self.fuse_pixel(d)
        logit_pixel = self.logit_pixel(fuse_pixel)

        e = F.adaptive_avg_pool2d(e4, output_size=1).view(batch_size, -1)
        e = F.dropout(e, p=0.50, training=self.training)  # 16, 512
        fuse_image = self.fuse_image(e)  # 16, 64
        logit_image = self.logit_image(fuse_image).view(-1)  # 16

        fuse = self.fuse(
            torch.cat([
                fuse_pixel,
                F.upsample(
                    fuse_image.view(batch_size, -1, 1, 1),
                    scale_factor=128,
                    mode='nearest')
            ], 1))  # 16, 64, 128, 128
        logit = self.logit(fuse)  # 16, 1, 128, 128

        return logit, logit_pixel, logit_image

    def criterion(self,
                  logit_fuse,
                  logit_pixel,
                  logit_image,
                  truth_pixel,
                  truth_image,
                  is_average=True):
        loss_fuse = lovasz_loss(
            logit_fuse, truth_pixel, per_image=True, is_average=True)

        loss_pixel = lovasz_loss(
            logit_pixel, truth_pixel, per_image=True, is_average=False)
        # loss_pixel = FocalLoss(
        #     gamma=2, is_average=False)(logit_pixel, truth_pixel)

        # loss_fuse = loss_fuse * truth_image  # loss for empty image is weighted 0
        # if is_average:
        #     loss_fuse = loss_fuse.sum() / (truth_image.sum() + 1e-6)

        loss_pixel = loss_pixel * truth_image
        if is_average:
            loss_pixel = loss_pixel.sum() / (truth_image.sum() + 1e-6)

        loss_image = F.binary_cross_entropy_with_logits(
            logit_image, truth_image, reduce=is_average)

        weight_fuse, weight_pixel, weight_image = 1, 0.5, 0.05

        return weight_fuse * loss_fuse, weight_pixel * loss_pixel, weight_image * loss_image


class LinkNetAttention(nn.Module):
    def __init__(self,
                 num_classes=1,
                 pretrained=True,
                 nonlocal_mode='concatenation',
                 attention_dsample=2,
                 is_batchnorm=True):
        super(LinkNetAttention, self).__init__()
        self.is_batchnorm = is_batchnorm
        self.num_classes = num_classes
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=pretrained)

        self.pool = nn.MaxPool2d(2, 2)

        # downsampling
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                      resnet.maxpool)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = UnetConv2(filters[3], filters[3], self.is_batchnorm)
        self.gating = UnetGridGatingSignal(
            filters[3],
            filters[3],
            kernel_size=1,
            is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock4 = MultiAttentionBlock(
            in_size=filters[3],
            gate_size=filters[3],
            inter_size=filters[3],
            nonlocal_mode=nonlocal_mode,
            sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(
            in_size=filters[2],
            gate_size=filters[2],
            inter_size=filters[2],
            nonlocal_mode=nonlocal_mode,
            sub_sample_factor=attention_dsample)
        self.attentionblock2 = MultiAttentionBlock(
            in_size=filters[1],
            gate_size=filters[1],
            inter_size=filters[1],
            nonlocal_mode=nonlocal_mode,
            sub_sample_factor=attention_dsample)

        # Decoder
        self.decoder4 = DecoderBlockLinkNet(filters[3], filters[2])
        self.decoder3 = DecoderBlockLinkNet(filters[2], filters[1])
        self.decoder2 = DecoderBlockLinkNet(filters[1], filters[0])
        self.decoder1 = DecoderBlockLinkNet(filters[0], filters[0])

        # Final Classifier
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # # Gating Signal Generation
        center = self.center(self.pool(e4))
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder with Skip Connections)
        g_conv4, att4 = self.attentionblock4(e4, gating)
        d4 = self.decoder4(g_conv4) + e3
        g_conv3, att3 = self.attentionblock3(e3, d4)
        d3 = self.decoder3(g_conv3) + e2
        g_conv2, att2 = self.attentionblock2(e2, d3)
        d2 = self.decoder2(g_conv2) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        # f5 = self.finalconv3(f4)
        f5 = self.finalconv3(F.dropout2d(f4, p=0.2))

        if self.num_classes > 1:
            x_out = F.log_softmax(f5, dim=1)
        else:
            x_out = f5
        return x_out


# https://github.com/ozan-oktay/Attention-Gated-Networks
class UNetAttention(nn.Module):
    def __init__(self,
                 feature_scale=4,
                 n_classes=1,
                 is_deconv=True,
                 in_channels=3,
                 nonlocal_mode='concatenation',
                 attention_dsample=2,
                 is_batchnorm=True):
        super(UNetAttention, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UnetConv2(filters[3], filters[4], self.is_batchnorm)
        self.gating = UnetGridGatingSignal(
            filters[4],
            filters[4],  # filters[3]
            kernel_size=1,
            is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(
            in_size=filters[1],
            gate_size=filters[2],  # filters[3]
            inter_size=filters[1],
            nonlocal_mode=nonlocal_mode,
            sub_sample_factor=attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(
            in_size=filters[2],
            gate_size=filters[3],
            inter_size=filters[2],
            nonlocal_mode=nonlocal_mode,
            sub_sample_factor=attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(
            in_size=filters[3],
            gate_size=filters[4],  # filters[3]
            inter_size=filters[3],
            nonlocal_mode=nonlocal_mode,
            sub_sample_factor=attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # # Attention Mechanism
        # g_conv4, att4 = self.attentionblock4(conv4, gating)
        # g_conv3, att3 = self.attentionblock3(conv3, gating)
        # g_conv2, att2 = self.attentionblock2(conv2, gating)

        # # Upscaling Part (Decoder)
        # up4 = self.up_concat4(g_conv4, center)
        # up3 = self.up_concat3(g_conv3, up4)
        # up2 = self.up_concat2(g_conv2, up3)
        # up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

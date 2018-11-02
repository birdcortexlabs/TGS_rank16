#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int):
        super(ConvRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock_bak(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self,
                 in_channels,
                 middle_channels,
                 out_channels,
                 is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),  # nearest
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class DecoderBlockLinkNet(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C/4, 2 * H, 2 * W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        # x = self.relu(x)  # TODO undo
        return x


class ConvBn2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 dilation=1,
                 stride=1,
                 groups=1,
                 is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = SynchronizedBatchNorm2d(out_channels) ?
        if is_bn is False:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939
class Seq_Ex_Block(nn.Module):
    def __init__(self, in_ch, r=16):
        super(Seq_Ex_Block, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch),
            nn.Sigmoid())

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        # print(f'x:{x.sum()}, x_se:{x.mul(se_weight).sum()}')
        return x.mul(se_weight)


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.view(*(x.shape[:-2]), -1).mean(-1)


# https://kaggle2.blob.core.windows.net/forum-message-attachments/265701/8204/se_resnet3.py
# http://xxuan.me/2018-07-13-image-classification.html
class SeScale(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SeScale, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class SpatialGate2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            # nn.Linear(in_channels, in_channels // reduction),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            # nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid())

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, output_size=1)
        y = self.fc(y)

        # n_batches, n_channels, _, _ = x.size()
        # y = F.adaptive_avg_pool2d(x, output_size=1).view(n_batches, n_channels)
        # y = self.fc(y).view(n_batches, n_channels, 1, 1)

        return x * y


class ChannelGate2d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        y = self.fc(x)

        return x * y


class DecoderHyper2(nn.Module):
    def __init__(self, channel, in_channels, middle_channels, out_channels):
        super().__init__()
        self.dconv = nn.ConvTranspose2d(
            channel, channel, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvBn2d(
            in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(
            middle_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = SpatialGate2d(out_channels)
        self.channel_gate = ChannelGate2d(out_channels)

    def forward(self, x, e=None):
        x = F.relu(self.dconv(x), inplace=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 * x + g2 * x

        return x


class DecoderHyper(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBn2d(
            in_channels, middle_channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(
            middle_channels, out_channels, kernel_size=3, padding=1)
        self.spatial_gate = SpatialGate2d(out_channels)
        self.channel_gate = ChannelGate2d(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        g1 = self.spatial_gate(x)  # attention gate, (8, 64, 14, 14)
        g2 = self.channel_gate(x)  # (8, 64, 14, 14)
        x = g1 * x + g2 * x  # (8, 64, 14, 14)

        return x


class _GridAttentionBlockND(nn.Module):
    def __init__(self,
                 in_channels,
                 gating_channels,
                 inter_channels=None,
                 dimension=3,
                 mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in [
            'concatenation', 'concatenation_debug', 'concatenation_residual'
        ]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplementedError

        # Output transform
        self.W = nn.Sequential(
            conv_nd(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=self.sub_sample_kernel_size,
            stride=self.sub_sample_factor,
            padding=0,
            bias=False)
        self.phi = conv_nd(
            in_channels=self.gating_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.psi = conv_nd(
            in_channels=self.inter_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        # Initialise weights
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(
            self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(
            sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(
            self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(
            sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(
            self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(
            f, dim=2).view(batch_size, 1,
                           *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(
            sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock2D(_GridAttentionBlockND):
    def __init__(self,
                 in_channels,
                 gating_channels,
                 inter_channels=None,
                 mode='concatenation',
                 sub_sample_factor=2):
        super(GridAttentionBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            gating_channels=gating_channels,
            dimension=2,
            mode=mode,
            sub_sample_factor=sub_sample_factor,
        )


class UnetConv2(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 is_batchnorm,
                 n=2,
                 ks=3,
                 stride=1,
                 padding=1):
        super(UnetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.BatchNorm2d(out_size),
                    nn.ReLU(inplace=True),
                )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, ks, s, p),
                    nn.ReLU(inplace=True),
                )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(UnetUp, self).__init__()
        self.conv = UnetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(
                in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1:
                continue
            # init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetGridGatingSignal(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=1, is_batchnorm=True):
        super(UnetGridGatingSignal, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size, 1, 0),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size, 1, 0),
                nn.ReLU(inplace=True),
            )

        # # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode,
                 sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(
            in_channels=in_size,
            gating_channels=gate_size,
            inter_channels=inter_size,
            mode=nonlocal_mode,
            sub_sample_factor=sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock2D(
            in_channels=in_size,
            gating_channels=gate_size,
            inter_channels=inter_size,
            mode=nonlocal_mode,
            sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(
            nn.Conv2d(
                in_size * 2, in_size, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock2D') != -1:
                continue
            # init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat(
            [attention_1, attention_2], 1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        nn.init.kaiming_normal(m, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)

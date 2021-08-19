import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class DownBlock(nn.Module):
    def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = opt.negval

        if nFeat is None:
            nFeat = opt.n_feats

        if in_channels is None:
            in_channels = opt.n_colors

        if out_channels is None:
            out_channels = opt.n_colors

        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=False, act=nn.ReLU()):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

# class Upsampler(nn.Sequential):
#     def __init__(self, conv, scale, n_feat, bn=False, act=nn.ReLU(), bias=True):
#
#         m = []
#         if scale == 2:    # Is scale = 2^n?
#             m.append(conv(n_feat, 4 * n_feat, 3, bias))
#             m.append(nn.PixelShuffle(2))
#             if bn: m.append(nn.BatchNorm2d(n_feat))
#             m.append(act)
#         elif scale == 3:
#             m.append(conv(n_feat, 9 * n_feat, 3, bias))
#             m.append(nn.PixelShuffle(3))
#             if bn: m.append(nn.BatchNorm2d(n_feat))
#             m.append(act)
#         else:
#             raise NotImplementedError
#
#         super(Upsampler, self).__init__(*m)
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class disentangle(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act=nn.ReLU()):
        super(disentangle, self).__init__()
        predict_t = []
        for i in range(4):
            predict_t.append(RCAB(conv, n_feats, kernel_size, act=act))
        self.predict_t = nn.Sequential(*predict_t)

        predict_a = []
        for i in range(2):
            predict_a.append(RCAB(conv, n_feats, kernel_size, act=act))
        self.predict_a = nn.Sequential(*predict_a)

        predict_j = []
        for i in range(6):
            predict_j.append(RCAB(conv, n_feats, kernel_size, act=act))
        self.predict_j = nn.Sequential(*predict_j)

        self.conv_t = conv(n_feats, 3, kernel_size)
        self.conv_a = conv(n_feats, 3, kernel_size)
        self.conv_j = conv(n_feats, 3, kernel_size)
    def forward(self, x):
        t = self.predict_t(x)
        a = self.predict_a(x)
        j = self.predict_j(x)
        t = self.conv_t(t)
        a = self.conv_a(a)
        dehaze = self.conv_j(j)
        shape_out = a.data.size()
        a = F.avg_pool2d(a, shape_out[2])
        haze = dehaze*t + a*(1-t)
        return haze, dehaze, t, a


class decoder(nn.Module):
    def __init__(self, conv, in_feats, kernel_size, blocks, act=nn.ReLU()):
        super(decoder, self).__init__()
        up_blocks = [
            RCAB(
                conv, in_feats, kernel_size, act=act
            ) for _ in range(blocks)
        ]
        self.up_blocks = nn.Sequential(*up_blocks)
        self.up = Upsampler(conv, 2, in_feats)


    def forward(self, x):
        x = self.up_blocks(x)
        x = self.up(x)

        return x

class SALayer(nn.Module):
    def __init__(self, channel):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, 1, kernel_size=1, padding=0, bias=True),
                nn.Sigmoid()
        )
        # self.conv_du = nn.Sequential(
        #     nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
        #     nn.Sigmoid()
        # )
    def forward(self, x):
        y = self.conv_du(x)
        return x * y

class RSAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):
        super(RSAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(SALayer(n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, feats, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feats = feats

        self.inc = DoubleConv(n_channels, self.feats)
        self.down1 = Down(self.feats, self.feats * 2)
        self.down2 = Down(self.feats * 2, self.feats * 4)
        self.down3 = Down(self.feats * 4, self.feats * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.feats * 8, self.feats * 16 // factor)
        self.up1 = Up(self.feats * 16, self.feats * 8 // factor, bilinear)
        self.up2 = Up(self.feats * 8, self.feats * 4 // factor, bilinear)
        self.up3 = Up(self.feats * 4, self.feats * 2 // factor, bilinear)
        self.up4 = Up(self.feats * 2, self.feats, bilinear)
        self.outc = OutConv(self.feats, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output
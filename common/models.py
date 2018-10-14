# -*- coding: utf-8 -*-
# Author: Ji Yang <jiyang.py@gmail.com>
# License: MIT

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from common import lovasz_hinge


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  )

    def forward(self, x):
        return self.conv(x)


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)


class HengDecoder(nn.Module):
    """Decoder block adopted from CherKeng's implementation.

    Combined hypercolumn, scSE module with .
    """

    def __init__(self, in_channels, channels, out_channels):
        super(HengDecoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.scSEBlock = SCSEBlock(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        x = self.scSEBlock(x)
        return x


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16, ordinary=True):
        super(SCSEBlock, self).__init__()
        self.ordinary = ordinary
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        if self.ordinary:
            return chn_se + spa_se
        else:
            return torch.mul(torch.mul(x, chn_se), spa_se)


class UNetResNet(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    """

    def __init__(self, encoder_depth, dropout_2d=0.5,
                 pretrained=True):
        super().__init__()
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = models.resnet34(pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = models.resnet101(pretrained=pretrained)
        elif encoder_depth == 152:
            self.encoder = models.resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError('only 34, 101, 152 version of ResNet are implemented')

        self.conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu
        )  # 64
        self.encoder2 = self.encoder.layer1  # 64
        self.encoder3 = self.encoder.layer2  # 128
        self.encoder4 = self.encoder.layer3  # 256
        self.encoder5 = self.encoder.layer4  # 512
        self.center = nn.Sequential(
            ConvBnRelu2d(512, 512),
            ConvBnRelu2d(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder5 = HengDecoder(512 + 512, 512, 64)
        self.decoder4 = HengDecoder(64 + 256, 256, 64)
        self.decoder3 = HengDecoder(64 + 128, 128, 64)
        self.decoder2 = HengDecoder(64 + 64, 64, 64)
        self.decoder1 = HengDecoder(64, 32, 64)

        self.logit = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # batch_size, C, H, W = x.shape
        x = self.conv1(x)
        e2 = self.encoder2(x)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        f = self.center(e5)

        d5 = self.decoder5(f, e5)
        d5 = F.dropout2d(d5, p=0.05)
        d4 = self.decoder4(d5, e4)
        d4 = F.dropout2d(d4, p=0.05)
        d3 = self.decoder3(d4, e3)
        d3 = F.dropout2d(d3, p=0.1)
        d2 = self.decoder2(d3, e2)
        d2 = F.dropout2d(d2, p=0.1)
        d1 = self.decoder1(d2)

        # Hypercolumn
        f = torch.cat([
            d1,
            F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
            F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
            F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
            F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
        ], 1)

        f = F.dropout2d(f, p=self.dropout_2d)
        logit = self.logit(f)

        return logit

    def criterion(self, logit, truth):
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)
        loss = lovasz_hinge(logit, truth, per_image=True, ignore=None)
        return loss

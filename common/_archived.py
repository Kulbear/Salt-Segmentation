# -*- coding: utf-8 -*-
# Author: Ji Yang <jiyang.py@gmail.com>
# License: MIT

import torch
from torch import nn
from torch.nn import functional as F
from .models import ConvBn2d
from .lovasz_loss import lovasz_hinge


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.down1 = nn.Sequential(
            ConvBn2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            ConvBn2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            ConvBn2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down4 = nn.Sequential(
            ConvBn2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down5 = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.center = nn.Sequential(
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up5 = nn.Sequential(
            ConvBn2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            ConvBn2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            ConvBn2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            ConvBn2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            ConvBn2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.feature = nn.Sequential(
            ConvBn2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.logit = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        down1 = self.down1(x)
        f = F.max_pool2d(down1, kernel_size=2, stride=2)
        down2 = self.down2(f)
        f = F.max_pool2d(down2, kernel_size=2, stride=2)
        down3 = self.down3(f)
        f = F.max_pool2d(down3, kernel_size=2, stride=2)
        down4 = self.down4(f)
        f = F.max_pool2d(down4, kernel_size=2, stride=2)
        down5 = self.down5(f)
        f = F.max_pool2d(down5, kernel_size=2, stride=2)

        f = self.center(f)

        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up5(torch.cat([down5, f], 1))
        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up4(torch.cat([down4, f], 1))
        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up3(torch.cat([down3, f], 1))
        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up2(torch.cat([down2, f], 1))
        f = F.upsample(f, scale_factor=2, mode='bilinear', align_corners=True)
        f = self.up1(torch.cat([down1, f], 1))

        f = self.feature(f)
        logit = self.logit(f)

        return logit

    def criterion(self, logit, truth):
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)
        loss = lovasz_hinge(logit, truth, per_image=True, ignore=None)
        return loss

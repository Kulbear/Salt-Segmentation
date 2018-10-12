# -*- coding: utf-8 -*-
# Author: Ji Yang <jiyang.py@gmail.com>
# License: MIT

import torch.nn as nn
import torch.nn.functional as F

SMOOTH = 1e-7


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, eps=1e-8):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        # print(input.size(), target.size())
        assert input.size() == target.size()

        # input = input.transpose(1,2).transpose(2,3).contiguous().view(-1,1)
        # target = target.transpose(1,2).transpose(2,3).contiguous().view(-1,1)

        p = input.sigmoid()
        p = p.clamp(min=self.eps, max=1. - self.eps)

        pt = p * target + (1. - p) * (1 - target)
        # from Heng, don't apply focal weight to predictions with prob < 0.1
        # pt[pt < 0.1] = 0.

        w = (1. - pt).pow(self.gamma)

        loss = F.binary_cross_entropy_with_logits(input, target, w)

        return loss


def accuracy(prediction, truth, threshold=0.5):
    """Calculate the accuracy of the prediction after thresholding the prediction."""

    # Here we expect `prediction` is a tensor contains probabilities
    # Usually from a Sigmoid or Softmax output
    prediction = (prediction > threshold).float().view(-1)
    truth = truth.view(-1)
    n_all = truth.size()[0]
    return (prediction == truth).sum().float() / n_all


# Taken from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
# Author: Ilya Ezepov (https://www.kaggle.com/iezepov)
def iou(prediction, truth, is_averaged=True):
    """Calculate the Intersection over Union score."""

    prediction = (prediction > 0.5).int().view(-1)
    truth = truth.int().view(-1)
    intersection = (prediction & truth).float().sum()
    union = (prediction | truth).float().sum()

    iou = intersection / (union + SMOOTH)  # Handle zero division

    return iou.mean() if is_averaged else iou

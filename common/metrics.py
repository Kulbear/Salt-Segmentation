# -*- coding: utf-8 -*-
# Author: Ji Yang <jiyang.py@gmail.com>
# License: MIT

SMOOTH = 1e-7


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

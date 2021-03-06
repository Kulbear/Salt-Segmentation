# Lovasz-Softmax and Jaccard hinge loss in PyTorch
# Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)

from itertools import filterfalse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.

    Note: see Alg. 1 in the paper for details.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum().float()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """Calculate IoU for the foreground class.

    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)  # mean across images if per_image
    return 100 * iou


def _iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """Array of IoU for each (non ignored) class."""
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious))  # mean across images if per_image
    return 100 * np.array(ious)


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """Calculate binary Lovasz hinge loss.

        logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """Calculate binary Lovasz hinge loss.

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
    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """Flatten predictions in the batch (binary case).

    Note: remove labels equal to 'ignore'.
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    v_scores = scores[valid]
    v_labels = labels[valid]
    return v_scores, v_labels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_ce_loss(logits, labels, ignore=None):
    """Calculate binary cross-entropy loss.

        logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """Calculate multi-class Lovasz-Softmax loss.

        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        only_present: average only on classes present in ground truth
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
    """
    if per_image:
        loss = mean(
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
            for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """Calculate multi-class Lovasz-Softmax loss.

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

        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """Flatten predictions in the batch."""
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    v_probas = probas[valid.nonzero().squeeze()]
    v_labels = labels[valid]
    return v_probas, v_labels


def ce_loss(logits, labels):
    """Calculate cross-entropy loss."""
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


def mean(l, ignore_nan=False, empty=0):
    """Nan mean compatible with generators."""
    l = iter(l)
    if ignore_nan:
        l = filterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

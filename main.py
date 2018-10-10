# -*- coding: utf-8 -*-
# Author: Ji Yang <jiyang.py@gmail.com>
# License: MIT

import numpy as np

from salt_dataset import SaltDataset
from salt_parser import SaltParser
from common import UNetResNet

from util import process, get_train_val

from train import train_stage1
from train import train_stage2

import torch
import torch.optim as optim

import warnings

warnings.filterwarnings("ignore")

device = 'cuda'

use_depth = False

# Data loading
sp = SaltParser()
sp.load_data()
sp.train_df['X_train'] = list(sp.X_train)
sp.train_df['y_train'] = list(sp.y_train)
train_df = sp.train_df

n_fold = 5
train_df.sort_values('coverage_class', inplace=True)
train_df['fold'] = (list(range(n_fold)) * train_df.shape[0])[:train_df.shape[0]]
subsets = [train_df[train_df['fold'] == i] for i in range(n_fold)]

# stage 1

stage1_stats = []

for fold_idx in range(n_fold):
    # get the train/val split
    X_tr, X_val, y_tr, y_val = get_train_val(subsets, fold_idx)
    # add depth information or triple the gray channel
    X_tr, y_tr = process(X_tr, y_tr, use_depth=use_depth)
    X_val, y_val = process(X_val, y_val, use_depth=use_depth)
    y_tr = np.squeeze(y_tr)
    y_val = np.squeeze(y_val)

    # prepare PyTorch dataset and dataloader
    train_ds = SaltDataset(X_tr, y_tr)
    val_ds = SaltDataset(X_val, y_val, transform=None)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=True)

    net = UNetResNet(34)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    net, stat = train_stage1(net, optimizer, train_loader, val_loader, fold=fold_idx)
    stage1_stats.append(stat)

# a = []
# for s in stage1_stats:
#     a.append(np.array(s))
# np.squeeze(np.array(a))

# stage 2

for fold_idx in range(n_fold):
    # get the train/val split
    X_tr, X_val, y_tr, y_val = get_train_val(subsets, fold_idx)
    # add depth information or triple the gray channel
    X_tr, y_tr = process(X_tr, y_tr, use_depth=use_depth)
    X_val, y_val = process(X_val, y_val, use_depth=use_depth)
    y_tr = np.squeeze(y_tr)
    y_val = np.squeeze(y_val)

    # prepare PyTorch dataset and dataloader
    train_ds = SaltDataset(X_tr, y_tr)
    val_ds = SaltDataset(X_val, y_val, transform=None)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=32, shuffle=True)

    net = UNetResNet(34)
    net.to(device)

    train_stage2(net, train_loader, val_loader, fold=fold_idx)

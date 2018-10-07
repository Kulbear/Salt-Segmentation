# -*- coding: utf-8 -*-
# Author: Ji Yang <jiyang.py@gmail.com>
# License: MIT

from collections import Counter

import pandas as pd
import numpy as np

IMAGE_SIZE = (101, 101)

convert_1d = lambda x: np.array(x.tolist()).reshape(-1, *IMAGE_SIZE, 1)
convert_3d = lambda x: np.array(x.tolist()).reshape(-1, *IMAGE_SIZE, 3)


def get_train_val(n_fold_data, fold_idx):
    train = pd.concat([_ for idx, _ in enumerate(n_fold_data) if idx != fold_idx])
    valid = n_fold_data[fold_idx]
    # print(Counter(train.coverage_class))
    # print(Counter(valid.coverage_class))
    x_train = convert_1d(train['X_train'])
    x_valid = convert_1d(valid['X_train'])
    y_train = convert_1d(train['y_train'])
    y_valid = convert_1d(valid['y_train'])
    return x_train, x_valid, y_train, y_valid


def process(X, y, use_depth=False):
    def add_channels(image, depth_included=False):
        result = np.zeros((3, *image.shape))
        pure_depth = np.zeros_like(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, 0)
        for i in range(1, 101 + 1):
            pure_depth[i - 1] = i
        image = image / 255
        result[0] = image
        if depth_included:
            result[1] = pure_depth / 101
            result[2] = pure_depth / 101 * image
        else:
            result[1] = image
            result[2] = image
        result = np.rollaxis(result, 0, 3)
        return result

    X_ = []
    y_ = []
    for i in range(X.shape[0]):
        img_temp, mask_temp = X[i], y[i]
        img_temp = add_channels(img_temp.reshape(101, 101), depth_included=use_depth)
        X_.append(img_temp)
        y_.append(mask_temp)

    X_ = np.asarray(X_)
    y_ = np.asarray(y_)

    return X_, y_

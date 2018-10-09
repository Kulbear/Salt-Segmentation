# -*- coding: utf-8 -*-
# Author: Ji Yang <jiyang.py@gmail.com>
# License: MIT

import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm


class SaltParser(object):
    """Parser for Salt Competition."""

    def __init__(self,
                 data_src='./input/',
                 load_test_data=True):

        self.data_src = data_src
        self.load_test_data = load_test_data

        self.train_df = None
        self.test_df = None

        self.X_train = []
        self.y_train = []

        self.initialize_data()

    def initialize_data(self):
        """Initialize processing by loading .csv files."""

        train_df = pd.read_csv('{}train.csv'.format(self.data_src),
                               usecols=[0], index_col='id')
        depths_df = pd.read_csv('{}depths.csv'.format(self.data_src),
                                index_col='id')

        self.train_df = train_df.join(depths_df)
        self.test_df = depths_df[~depths_df.index.isin(train_df.index)]

    def load_data(self):
        """Load images and masks from training set."""
        print('Loading training set.')
        # Loop over ids in train_df
        for i in tqdm(self.train_df.index):
            img_src = '{}train/images/{}.png'.format(self.data_src, i)
            mask_src = '{}train/masks/{}.png'.format(self.data_src, i)
            img_temp = cv2.imread(img_src, 0)
            # Load mask
            mask_temp = cv2.imread(mask_src, 0)

            self.X_train.append(img_temp)
            self.y_train.append(mask_temp)

        # Transform into arrays
        self.X_train = np.asarray(self.X_train)
        self.y_train = np.asarray(self.y_train)

        self.compute_coverage()

    def compute_coverage(self):
        """Compute salt coverage of each mask. 
        
        This will serve as a basis for stratified split between training and validation sets.
        """

        def cov_to_class(val):
            for i in range(0, 11):
                if val * 10 <= i:
                    return i

        # Output the percentage of area covered by class
        self.train_df['coverage'] = np.mean(self.y_train / 255., axis=(1, 2))
        # Coverage must be split into bins, otherwise stratified split will not be possible,
        # because each coverage will occur only once.
        self.train_df['coverage_class'] = self.train_df.coverage.map(cov_to_class)

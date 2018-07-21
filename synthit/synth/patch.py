#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.synth.patch

base class for patch-based synthesis routines

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

__all__ = ['PatchSynth']

import logging

import ants
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from ..errors import SynthError
from ..util.io import glob_nii
from ..util.patches import extract_patches

logger = logging.getLogger(__name__)


class PatchSynth():
    """
    provides the model for training and synthesizing MR neuro images via patch-based methods

    Args:
        regr (sklearn model): an instantiated model class (e.g., sklearn.ensemble.forest.RandomForestRegressor)
            needs to have a fit and predict public method
        patch_size (int): size of patch to use (patch_size x patch_size x patch_size)
        n_samples (int): number of patches (i.e., samples) to use from each image
        context_radius (tuple): tuple containing number of voxels away to get context from (e.g., (3,5) means
            get context values at 3 voxels and 5 voxels away from the patch center)
        threshold (float): threshold that separated background and foreground (foreground greater than threshold)
        poly_deg (int): degree of polynomial features to generate from patch samples
        mean (bool): use the mean of the patch instead of the patch values
        full_patch (bool): use a full patch instead of the 6-nearest neighbors
        flatten (bool): flatten the target voxel intensities (needed in some types of regressors)
    """

    def __init__(self, regr, patch_size=3, n_samples=1e5, context_radius=(3,5,7), threshold=0, poly_deg=None,
                 mean=False, full_patch=False, flatten=True):
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.context_radius = context_radius
        self.threshold = threshold
        self.poly_deg = poly_deg
        self.mean = mean
        self.economy_patch = not full_patch
        self.flatten = flatten
        self.regr = regr

    def fit(self, source, target, mask=None):
        """ train the model for synthesis given a set of source and target images """
        X, y = self.extract_patches_train(source, target, mask)
        if self.poly_deg is not None:
            logger.info('Creating polynomial features')
            poly = PolynomialFeatures(self.poly_deg)
            X = poly.fit_transform(X)
        logger.info('Training the model')
        self.regr.fit(X, y.flatten() if self.flatten else y)

    def predict(self, source, mask=None):
        """ synthesize/predict an image from a source (input) image """
        logger.info('Extracting patches')
        X, idxs = self.extract_patches_predict(source, mask)
        if self.poly_deg is not None:
            logger.info('Creating polynomial features')
            poly = PolynomialFeatures(self.poly_deg)
            X = poly.fit_transform(X)
        logger.info('Starting synthesis')
        y = self.regr.predict(X)
        synthesized = source[0].numpy()
        msk = np.zeros(synthesized.shape, dtype=bool)
        msk[idxs] = True
        synthesized[msk] = y.flatten()
        synthesized[~msk] = np.min(y)-1
        predicted = source[0].new_image_like(synthesized)
        return predicted

    def extract_patches_train(self, source, target, mask=None):
        """ get patches and corresponding target voxel intensity values for training """
        all_patches = []
        all_out = []
        if any([len(source_) != len(target) for source_ in source]) or len(target) == 0:
            raise SynthError('Number of source and target images must be the same in training and non-zero!')
        mask = [None] * len(target) if mask is None else mask
        for i, (*src, tgt, msk) in enumerate(zip(zip(*source), target, mask), 1):
            src = src[0]  # extract the tuple since it is currently inside a list
            logger.info('Extracting patches ({:d}/{:d})'.format(i, len(target)))
            src_data = [src_.numpy() for src_ in src]
            tgt_data = tgt.numpy()
            # only use the first for consistency across indices since co-registration assumed
            idxs = np.where(src_data[0] > self.threshold) if msk is None else np.where(msk.numpy() == 1)
            if self.n_samples is not None:
                if idxs[0].size < self.n_samples:
                    logger.warning('n_samples is greater than the number of samples available in the image ({} > {})'
                                   .format(self.n_samples, idxs[0].size))
                choices = np.random.choice(np.arange(idxs[0].size), int(self.n_samples), replace=True)
                idxs = tuple([idx[choices] for idx in idxs])
            if len(source) == 1:
                patches = self.__extract_patches(src_data[0], idxs)
            else:
                patches = np.hstack([self.__extract_patches(src_data_, idxs) for src_data_ in src_data]).squeeze()
            out = tgt_data[idxs][:, np.newaxis] if not self.mean else self.__extract_patches(tgt_data, idxs)
            all_patches.append(patches)
            all_out.append(out)
        all_patches = np.vstack(all_patches)
        all_out = np.vstack(all_out)
        return all_patches, all_out

    def extract_patches_predict(self, source, mask=None):
        """ extract patches and get indices for prediction/synthesis """
        src_data = [src.numpy() for src in source]
        idxs = np.where(src_data[0] > self.threshold) if mask is None else np.where(mask.numpy() == 1)
        if len(source) == 1:
            patches = self.__extract_patches(src_data[0], idxs)
        else:
            patches = np.hstack([self.__extract_patches(src_data_, idxs) for src_data_ in src_data]).squeeze()
        return patches, idxs

    def __extract_patches(self, data, idxs):
        """ convenience wrapper for extract patches specific to this instantiation """
        patches = extract_patches(data, idxs, self.patch_size, self.threshold, self.context_radius,
                                  self.economy_patch, self.mean)
        return patches

    @staticmethod
    def image_list(img_dir):
        """ convenience function to get a list of images in ANTsImage format """
        img_fns = glob_nii(img_dir)
        imgs = [ants.image_read(img_fn) for img_fn in img_fns]
        return imgs

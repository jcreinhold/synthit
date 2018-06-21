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
        stride (int): number of patches to use from each image (use 1/stride patches from each image)
        context_radius (tuple): tuple containing number of voxels away to get context from (e.g., (3,5) means
            get context values at 3 voxels and 5 voxels away from the patch center)
        threshold (float): threshold that separated background and foreground (foreground greater than threshold)
        flatten (bool): flatten the target voxel intensities (needed in some types of regressors)
    """

    def __init__(self, regr, patch_size=3, stride=1, context_radius=(3,5,7), threshold=0, flatten=True):
        self.patch_size = patch_size
        self.stride = stride
        self.context_radius = context_radius
        self.threshold = threshold
        self.flatten = flatten
        self.regr = regr

    def fit(self, source, target, mask=None):
        """ train the model for synthesis given a set of source and target images """
        X, y = self.extract_patches_train(source, target, mask)
        logger.info('Training the model')
        self.regr.fit(X, y.flatten() if self.flatten else y)

    def predict(self, source, mask=None):
        """ synthesize/predict an image from a source (input) image """
        logger.info('Extracting patches')
        X, idxs = self.extract_patches_predict(source, mask)
        logger.info('Starting synthesis')
        y = self.regr.predict(X)
        synthesized = source.numpy()
        synthesized[idxs] = y.flatten()
        predicted = source.new_image_like(synthesized)
        return predicted

    def extract_patches_train(self, source, target, mask=None):
        """ get patches and corresponding target voxel intensity values for training """
        all_patches = []
        all_out = []
        if len(source) != len(target):
            raise SynthError('Number of source and target images must be the same in training!')
        mask = [None] * len(source) if mask is None else mask
        for i, (src, tgt, msk) in enumerate(zip(source, target, mask), 1):
            logger.info('Extracting patches ({:d}/{:d})'.format(i, len(source)))
            src_data = src.numpy()
            tgt_data = tgt.numpy()
            idxs = np.where(src_data > self.threshold) if msk is None else np.where(msk.numpy() == 1)
            idxs = [idx[::self.stride] for idx in idxs]
            patches = extract_patches(src_data, idxs, patch_size=self.patch_size, ctx_radius=self.context_radius)
            out = tgt_data[idxs][:,np.newaxis]
            all_patches.append(patches)
            all_out.append(out)
        all_patches = np.vstack(all_patches)
        all_out = np.vstack(all_out)
        return all_patches, all_out

    def extract_patches_predict(self, source, mask=None):
        """ extract patches and get indices for prediction/synthesis """
        src_data = source.numpy()
        idxs = np.where(src_data > self.threshold) if mask is None else np.where(mask.numpy() == 1)
        patches = extract_patches(src_data, idxs, patch_size=self.patch_size, ctx_radius=self.context_radius)
        return patches, idxs

    @staticmethod
    def image_list(img_dir):
        """ convenience function to get a list of images in ANTsImage format """
        img_fns = glob_nii(img_dir)
        imgs = [ants.image_read(img_fn) for img_fn in img_fns]
        return imgs

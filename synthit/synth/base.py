#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.synth.base

base class for synthesizers

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

__all__ = ['Synth']

import logging

import ants
import numpy as np

from ..errors import SynthError
from ..util.io import glob_nii
from ..util.patches import extract_patches

logger = logging.getLogger(__name__)


class Synth():

    def __init__(self, patch_size=3, stride=1, context_radius=7, min_val=0, n_jobs=-1):
        self.patch_size = patch_size
        self.stride = stride
        self.context_radius = context_radius
        self.min_val = min_val
        self.n_jobs = n_jobs

    def extract_patches_train(self, source, target):
        all_patches = []
        all_out = []
        if len(source) != len(target):
            raise SynthError('Number of source and target images must be the same in training!')
        for i, (src, tgt) in enumerate(zip(source, target), 1):
            logger.info('Extracting patches ({:d}/{:d})'.format(i, len(source)))
            src_data = src.numpy()
            tgt_data = tgt.numpy()
            idxs = np.where(src_data > self.min_val)
            idxs = [idxs[::self.stride] for idxs in idxs]
            patches = extract_patches(src_data, idxs, patch_size=self.patch_size, ctx_radius=self.context_radius)
            out = tgt_data[idxs][:,np.newaxis]
            all_patches.append(patches)
            all_out.append(out)
        all_patches = np.vstack(all_patches)
        all_out = np.vstack(all_out)
        return all_patches, all_out

    def extract_patches_predict(self, source):
        src_data = source.numpy()
        idxs = np.where(src_data > self.min_val)
        patches = extract_patches(src_data, idxs, patch_size=self.patch_size, ctx_radius=self.context_radius)
        return patches, idxs

    @staticmethod
    def image_list(path):
        """ convenience function to get a list of images in ANTsImage format """
        img_fns = glob_nii(path)
        imgs = [ants.image_read(img_fn) for img_fn in img_fns]
        return imgs

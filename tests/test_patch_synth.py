#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_patch_synth

test patch-based synthesis functions for common runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 01, 2018
"""

import os
import unittest

import nibabel as nib
from sklearn.linear_model import LinearRegression

from synthit import PatchSynth


class TestPatchSynth(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.img_fn = os.path.join(self.data_dir, 'test.nii.gz')
        self.mask_fn = os.path.join(self.mask_dir, 'mask.nii.gz')
        self.img = nib.load(self.img_fn)
        self.mask = nib.load(self.mask_fn)
        self.regr = LinearRegression()

    def test_patch_synth_default(self):
        ps = PatchSynth(self.regr, n_samples=10, flatten=False)
        ps.fit([[self.img]], [self.img], [self.mask])
        _ = ps.predict([self.img], self.mask)

    def test_patch_synth_one_sample(self):
        ps = PatchSynth(self.regr, patch_size=1, n_samples=10, context_radius=(0,), flatten=False)
        ps.fit([[self.img]], [self.img], [self.mask])
        _ = ps.predict([self.img], self.mask)

    def test_patch_synth_neighbors_only(self):
       ps = PatchSynth(self.regr, patch_size=0, n_samples=10, context_radius=(1,), flatten=False)
       ps.fit([[self.img]], [self.img], [self.mask])
       _ = ps.predict([self.img], self.mask)

    def test_patch_mlr(self):
        from synthit.synth.mlr import LinearRegressionMixture
        regr = LinearRegressionMixture(3, num_restarts=2, num_workers=1, max_iterations=1)
        ps = PatchSynth(regr, n_samples=10, flatten=True)
        ps.fit([[self.img]], [self.img], [self.mask])
        _ = ps.predict([self.img], self.mask)

    def tearDown(self):
        del self.img, self.mask


if __name__ == '__main__':
    unittest.main()

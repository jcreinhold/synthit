#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_utilities

test the functions located in util submodule for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: May 01, 2018
"""

import os
import unittest

import ants

from synthit import split_filename, glob_nii, extract_patches


class TestUtilities(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.img_fn = os.path.join(self.data_dir, 'test.nii.gz')
        self.mask_fn = os.path.join(self.mask_dir, 'mask.nii.gz')
        self.img = ants.image_read(self.img_fn)
        self.mask = ants.image_read(self.mask_fn)

    def test_glob_nii(self):
        fn = glob_nii(self.data_dir)[0]
        self.assertEqual(fn, self.img_fn)

    def test_split_filename(self):
        directory, fn, ext = split_filename(self.img_fn)
        self.assertEqual(directory, self.data_dir)
        self.assertEqual(fn, 'test')
        self.assertEqual(ext, '.nii.gz')

    def test_extract_patches(self):
        _ = extract_patches(self.img.numpy() * self.mask.numpy())

    def tearDown(self):
        del self.img, self.mask


if __name__ == '__main__':
    unittest.main()

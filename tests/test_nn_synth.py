#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_nn_synth

test neural network-based synthesis functions for common runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 07, 2018
"""

import os
import unittest

from synthit import NiftiImageDataset, RandomCrop
from synthit.models.nconvnet import Conv3dNLayerNet
from synthit.models.unet import Unet


class TestNNSynth(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')

    @unittest.skip("Not implemented.")
    def test_nn_synth_nconvnet(self):
        pass

    @unittest.skip("Not implemented.")
    def test_nn_synth_unet(self):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()

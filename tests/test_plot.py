#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_plot

test plotting functions for common runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 07, 2018
"""

import os
import shutil
import tempfile
import unittest

from synthit import directory_view, plot_dir_synth_quality


class TestPlot(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.out_dir = tempfile.mkdtemp()

    @unittest.skipIf("TRAVIS" in os.environ and os.environ["TRAVIS"] == "true", "Skipping this test on Travis CI.")
    def test_directory_view(self):
        directory_view(self.data_dir, out_dir=self.out_dir, trim=False)

    def test_synth_quality(self):
        plot_dir_synth_quality(self.data_dir, self.data_dir, mask_dir=self.mask_dir, out_dir=self.out_dir)

    def test_synth_quality_mean(self):
        plot_dir_synth_quality(self.data_dir, self.data_dir, mask_dir=self.mask_dir, out_dir=self.out_dir, mean=True)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()

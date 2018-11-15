#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_exec

test the synthit command line interfaces for runtime errors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 07, 2018
"""

import os
import shutil
import tempfile
import unittest

from synthit.exec.synth_train import main as synth_train
from synthit.exec.synth_predict import main as synth_predict


class TestCLI(unittest.TestCase):

    def setUp(self):
        wd = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(wd, 'test_data', 'images')
        self.mask_dir = os.path.join(wd, 'test_data', 'masks')
        self.out_dir = tempfile.mkdtemp()
        self.train_args = f'-s {self.data_dir} -t {self.data_dir}'.split()
        self.predict_args = f'-s {self.data_dir} -o {self.out_dir}/test'.split()

    def test_linear_regression_synth_cli(self):
        args = self.train_args + f'-m {self.mask_dir} -o {self.out_dir}/synth.pkl --n-samples 10 --patch-size 1 --ctx-radius 0 -r pr'.split()
        retval = synth_train(args)
        self.assertEqual(retval, 0)
        args = self.predict_args + f'-m {self.mask_dir} -t {self.out_dir}/synth.pkl'.split()
        retval = synth_predict(args)
        self.assertEqual(retval, 0)

    def tearDown(self):
        shutil.rmtree(self.out_dir)


if __name__ == '__main__':
    unittest.main()

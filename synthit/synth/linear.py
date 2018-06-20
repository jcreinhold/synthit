#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.linear

provides a linear regression model by which to do synthesis

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

__all__ = ['LinearSynth']

from sklearn import linear_model

from .base import Synth


class LinearSynth(Synth):

    def __init__(self, *args):
        super().__init__(*args)
        self.regr = linear_model.LinearRegression(n_jobs=self.n_jobs)

    def fit(self, source, target):
        X, y = self.extract_patches_train(source, target)
        self.regr.fit(X, y)

    def predict(self, source):
        X, idxs = self.extract_patches_predict(source)
        y = self.regr.predict(X)
        synthesized = source.numpy()
        synthesized[idxs] = y.flatten()
        predicted = source.new_image_like(synthesized)
        return predicted
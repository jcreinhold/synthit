#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.models.br

implements a bayesian regressor with pyro

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jul 21, 2018
"""

__all__ = ['BayesianRegression']

import logging

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
softplus = nn.Softplus()


class BayesianRegression:
    # TODO: how to incorporate knowledge of means of CSF, GM, WM in intensity normalized imgs
    def __init__(self, num_iterations=2000, num_predictions=100, lr=0.01):
        self.num_iterations = num_iterations
        self.num_predictions = num_predictions
        self.lr = lr
        self.N = None
        self.p = None
        self.regression_model = None

    def fit(self, X, y):
        self.N, self.p = X.shape
        self.regression_model = RegressionModel(self.p)
        X, y = torch.tensor(X).type(torch.Tensor), torch.tensor(y).type(torch.Tensor)
        data = torch.cat((X, y), 1)
        optim = Adam({"lr": self.lr})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        pyro.clear_param_store()
        for j in range(self.num_iterations):
            # calculate the loss and take a gradient step
            loss = svi.step(data)
            if j % 100 == 0:
                logger.info("[iteration %04d] loss: %.4f" % (j + 1, loss / float(self.N)))
        for name in pyro.get_param_store().get_all_param_names():
            logger.info("[{}]: {}".format(name, pyro.param(name).data.numpy()))

    def predict(self, X):
        x_data = torch.tensor(X).type(torch.Tensor)
        y_preds = torch.zeros(X.shape[0], 1)
        for _ in range(self.num_predictions):
            # guide does not require the data
            sampled_reg_model = self.guide(None)
            # run the regression model and add prediction to total
            y_preds = y_preds + sampled_reg_model(x_data)
        # take the average of the predictions
        y_preds = y_preds / self.num_predictions
        y_preds = y_preds.detach().numpy()
        return y_preds

    def model(self, data):
        # Create unit normal priors over the parameters
        loc = data.new_zeros(torch.Size((1, self.p)))
        scale = 2 * data.new_ones(torch.Size((1, self.p)))
        bias_loc = data.new_zeros(torch.Size((1,)))
        bias_scale = 2 * data.new_ones(torch.Size((1,)))
        w_prior = Normal(loc, scale).independent(1)
        b_prior = Normal(bias_loc, bias_scale).independent(1)
        priors = {'linear.weight': w_prior, 'linear.bias': b_prior}
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self.regression_model, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_module()

        with pyro.iarange("map", self.N, subsample=data):
            x_data = data[:, :-1]
            y_data = data[:, -1]
            # run the regressor forward conditioned on inputs
            prediction_mean = lifted_reg_model(x_data).squeeze(-1)
            pyro.sample("obs", Normal(prediction_mean, 1),
                        obs=y_data)

    def guide(self, data):
        w_loc = torch.randn(1, self.p)
        w_log_sig = torch.tensor(-3.0 * torch.ones(1, self.p) + 0.05 * torch.randn(1, self.p))
        b_loc = torch.randn(1)
        b_log_sig = torch.tensor(-3.0 * torch.ones(1) + 0.05 * torch.randn(1))
        # register learnable params in the param store
        mw_param = pyro.param("guide_mean_weight", w_loc)
        sw_param = softplus(pyro.param("guide_log_scale_weight", w_log_sig))
        mb_param = pyro.param("guide_mean_bias", b_loc)
        sb_param = softplus(pyro.param("guide_log_scale_bias", b_log_sig))
        # gaussian guide distributions for w and b
        w_dist = Normal(mw_param, sw_param).independent(1)
        b_dist = Normal(mb_param, sb_param).independent(1)
        dists = {'linear.weight': w_dist, 'linear.bias': b_dist}
        # overloading the parameters in the module with random samples from the guide distributions
        lifted_module = pyro.random_module("module", self.regression_model, dists)
        # sample a regressor
        return lifted_module()


class RegressionModel(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        return self.linear(x)

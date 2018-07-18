#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.util.mlr

holds code to fit and predict from a mixture of
linear regressors

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jul 18, 2018
"""

__all__ = ['LinearRegressionMixture']

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class LinearRegressionMixture:

    def __init__(self, K):
        self.K = K

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def __em(self, X, y):
        pass


def weighted_linear_regression(x, y, weights):
    """
    Perform linear regression and return the residuals. Note this assumes
    the weights are a vector rather than the standard diagonal matrix-- this is
    for computational efficiency.
    """
    return np.linalg.pinv((weights[:, np.newaxis] * x).T.dot(x)).dot((weights[:, np.newaxis] * x).T.dot(y))


def weighted_regression_variance(x, y, weights, coefficients):
    """Calculate the variance of a regression model where each observation is weighted."""
    # TODO: Vectorize
    result = 0.
    for i in range(len(y)):
        result += weights[i] * (y[i] - x[i].T.dot(coefficients)) ** 2
    return result / weights.sum()


def calculate_assignments(assignment_weights, stochastic):
    '''
    Assign each set of points to a component.
    If stochastic is true, randomly sample proportional to the assignment_weights.
    Otherwise, assign the component with the maximum weight.
    This is the C-step in the CEM algorithm.
    '''
    if stochastic:
        return np.array([np.random.choice(len(row), p=row) for row in assignment_weights])
    return np.argmax(assignment_weights, axis=1)


def calculate_assignment_weights(x, y, component_weights, coefficients, variances):
    """
    Determine a probability for each component to generate each point
    This is the E-step in the CEM algorithm
    """
    num_components = len(component_weights)

    # Initialize the new assignment weights
    assignment_weights = np.ones((len(x), num_components), dtype=float)

    # Calculate the likelihood of the points one at a time
    # to prevent underflow issues
    for xi, yi in zip(x, y):
        # Get the mean of each component
        mu = np.array([xi.dot(b) for b in coefficients])

        # Get the standard deviation of each component
        sigma = np.array([np.sqrt(v) for v in variances])

        # Calculate the likelihood of this data point coming from each component
        temp_weights = norm.pdf(yi, loc=mu, scale=sigma)

        # Update the likelihood of each component generating this set
        assignment_weights *= temp_weights / temp_weights.sum()
        assignment_weights /= assignment_weights.sum()

    # Multiply in the component weightings
    assignment_weights *= component_weights
    assignment_weights /= assignment_weights.sum()

    return assignment_weights


def maximum_likelihood_parameters(x, y, num_components, num_features, assignments, assignment_weights):
    """
    Calculate the parameter values that maximize the likelihood of the data.
    This is the M-step of the CEM algorithm.
    """
    # Calculate the weight of each component in the mixture
    component_weights = np.array([(assignments == i).sum() for i in range(num_components)]) / float(len(assignments))

    # Calculate the regression coefficients and variance for each component
    coefficients = np.zeros((num_components, num_features))
    variances = np.zeros(num_components)
    for i in xrange(num_components):
        # Get the points that are members of this component
        points = np.where(assignments == i)[0]

        # Get the weights for each set
        subset_weights = assignment_weights[points][:, i]

        # If no points were assigned to this cluster, soft-assign it random points
        # TODO: Is there a better way to proceed here? Some sort of split-merge type thing?
        if len(points) == 0:
            points = np.random.choice(len(assignments), size=np.random.randint(1, len(assignments)), replace=False)
            subset_weights = np.ones(len(points)) / float(len(points))

        # Get the data associated with this component
        component_x = []
        component_y = []
        weights = []
        for key, subset_weight in zip(keys[points], subset_weights):
            # Get the data for this subset
            x, y = data[key]

            # Add the points to the overall values to regress on
            component_x.extend(x)
            component_y.extend(y)

            # Each point in a set gets equal weight
            weights.extend([subset_weight / float(len(y))] * len(y))

        # Convert the results to numpy arrays
        component_x = np.array(component_x)
        component_y = np.array(component_y)
        weights = np.array(weights)

        # Get the weighted least squares coefficients
        coefficients[i] = weighted_linear_regression(component_x, component_y, weights)

        # Get the variance of the component given the coefficients
        variances[i] = weighted_regression_variance(component_x, component_y, weights, coefficients[i])

    return (component_weights, coefficients, variances)


def data_log_likelihood(data, keys, assignments, component_weights, coefficients, variances):
    '''
    Calculate the log-likelihood of the data being generated by the mixture model
    with the given parameters.
    '''
    log_likelihood = 0
    for i, key in enumerate(keys):
        x, y = data[key]

        assigned = assignments[i]

        mu = x.dot(coefficients[assigned])
        sigma = np.sqrt(variances[assigned])

        log_likelihood += np.log(norm.pdf(y, loc=mu, scale=sigma)).sum()
        log_likelihood += np.log(component_weights[assigned])

    return log_likelihood

def fit_mixture(data, keys, num_components, max_iterations, stochastic=False, verbose=False, threshold=0.00001):
    """
    Run the classification expecatation-maximization (CEM) algorithm to fit a maximum likelihood model.
    Note that the result is a local optimum, not necessarily a global one.
    """
    num_features = data.values()[0][0].shape[1]

    prev_log_likelihood = 1
    cur_log_likelihood = 0
    cur_iteration = 0

    # Random initialization
    assignment_weights = np.random.uniform(size=(len(data), num_components))
    assignment_weights /= assignment_weights.sum(axis=1)[:, np.newaxis]

    # Initialize using the normal steps now
    assignments = calculate_assignments(assignment_weights, True)

    component_weights, coefficients, variances = maximum_likelihood_parameters(data, keys, num_components, num_features,
                                                                               assignments, assignment_weights)

    while np.abs(prev_log_likelihood - cur_log_likelihood) > threshold and cur_iteration < max_iterations:

        # Calculate the expectation weights
        assignment_weights = calculate_assignment_weights(data, keys, component_weights, coefficients, variances)

        # Assign a value to each of the points
        assignments = calculate_assignments(assignment_weights, stochastic=stochastic)

        # Maximize the likelihood of the parameters
        component_weights, coefficients, variances = maximum_likelihood_parameters(data, keys, num_components,
                                                                                   num_features, assignments,
                                                                                   assignment_weights)

        # Calculate the total data log-likelihood
        prev_log_likelihood = cur_log_likelihood
        cur_log_likelihood = data_log_likelihood(data, keys, assignments, component_weights, coefficients, variances)

        # Add the iteration to the results
        results.add_iteration(assignments, component_weights, coefficients, variances, cur_log_likelihood)

        cur_iteration += 1

    # Tell the results that we're done fitting
    results.finish()

    return results

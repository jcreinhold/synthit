#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.util.mlr

holds code to fit and predict from a mixture of
linear regressors.

Note that this code is largely based off of [1]
(read: copied and modified) which is designed around [2]

References:
    [1] W. Tansey, https://github.com/tansey/regression_mixtures
    [2] S. Faria and G. Soromenho, “Fitting mixtures of linear regressions,”
        J. Stat. Comput. Simul., vol. 80, no. 2, pp. 201–225, 2010.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jul 18, 2018
"""

__all__ = ['LinearRegressionMixture']

import logging
from multiprocessing import Pool

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


class LinearRegressionMixture:
    """ Mixture of linear regressors model """

    def __init__(self, num_components, max_iterations=20, threshold=1e-10, num_restarts=1, num_workers=1, k=5):
        self.num_components = num_components
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.num_restarts = num_restarts
        self.num_workers = num_workers
        self.k = k

    def fit(self, X, y):
        if self.num_restarts > 1:
            results = fit_with_restarts(X, y, self.num_components, self.max_iterations, self.num_restarts, False, self.num_workers)
        else:
            results = fit_mixture(X, y, self.num_components, self.max_iterations, False, self.threshold)
        #del results.iterations  # delete this since it is large
        self.results = results
        self.component_weights = results.best.component_weights
        self.coefficients = results.best.coefficients
        self.variances = results.best.variances
        neigh = KNeighborsClassifier(self.k)
        self.classifier = neigh.fit(X, results.best.assignments)

    def predict(self, X):
        assignments = self.classifier.predict(X)
        y = np.zeros(X.shape[0])
        for i in range(self.num_components):
            y[assignments == i] = X[assignments == i] @ self.coefficients[i]
        return y


class MixtureModel:
    """Container to store the results of a single iteration of the CEM algorithm."""
    def __init__(self, assignments, component_weights, coefficients, variances):
        self.assignments = assignments
        self.component_weights = component_weights
        self.coefficients = coefficients
        self.variances = variances


class MixtureResults:
    """Container to store the results of a CEM iteration."""
    def __init__(self, num_components):
        self.num_components = num_components
        self.iterations = []
        self.log_likelihoods = []
        self.best = None

    def add_iteration(self, assignments, component_weights, coefficients, variances, data_log_likelihood):
        self.iterations.append(MixtureModel(assignments, component_weights, coefficients, variances))
        self.log_likelihoods.append(data_log_likelihood)

    def finish(self):
        """Tell the container we're done running CEM."""
        self.log_likelihoods = np.array(self.log_likelihoods)
        self.best = self.iterations[np.argmax(self.log_likelihoods)]


def weighted_linear_regression(X, y, weights):
    """
    Perform linear regression and return the residuals. Note this assumes
    the weights are a vector rather than the standard diagonal matrix-- this is
    for computational efficiency.

    Args:
        X (np.ndarray): source data (N x M matrix)
        y (np.ndarray): target data (N x 1 matrix)
        weights:

    Returns:
        wlr (np.ndarray): coefficients for mean in linear regression, i.e., beta (J x M)
    """
    W = weights[:, np.newaxis]
    wlr = np.linalg.pinv((W * X).T.dot(X)).dot((W * X).T.dot(y))
    return wlr


def weighted_regression_variance(X, y, weights, coefficients):
    """
    Calculate the variance of a regression model where each observation is weighted.

    Args:
        X (np.ndarray): source data (N x M matrix)
        y (np.ndarray): target data (N x 1 matrix)
        weights (np.ndarray): posterior probability
        coefficients: coefficients for mean, i.e., beta (J x M)

    Returns:
        wrv (np.ndarray): variances associated with components (J x 1)
    """
    wrv = ((weights * (y - X @ coefficients).T) @ (y - X @ coefficients)) / weights.sum()
    return wrv


def calculate_assignments(assignment_weights, stochastic=False):
    """
    Assign each set of points to a component.
    If stochastic is true, randomly sample proportional to the assignment_weights.
    Otherwise, assign the component with the maximum weight.
    This is the C-step in the CEM algorithm.

    Args:
        assignment_weights (np.ndarray): posterior prob for assignments
        stochastic (bool): randomly sample proportional to the assignment_weights

    Returns:
        assignments (np.ndarray): component assignment for all x,y pairs
    """
    if stochastic:
        assignments = np.array([np.random.choice(len(row), p=row) for row in assignment_weights])
    else:
        assignments = np.argmax(assignment_weights, axis=1)
    return assignments


def calculate_assignment_weights(X, y, component_weights, coefficients, variances):
    """
    Determine a probability for each component to generate each point
    This is the E-step in the CEM algorithm

    Args:
        X (np.ndarray): source data (N x M matrix)
        y (np.ndarray): target data (N x 1 matrix)
        component_weights (np.ndarray): prior probability (pi_j) (J x 1)
        coefficients (np.ndarray): coefficients for mean in linear regression, i.e., beta (J x M)
        variances (np.ndarray): variances associated with components (J x 1)

    Returns:
        assignment_weights (np.ndarray): posterior prob for assignments
    """
    num_components = len(component_weights)

    # Initialize the new assignment weights
    assignment_weights = np.ones((len(X), num_components), dtype=float)  # N x J

    # TODO: speed this up
    # Calculate the likelihood of the points one at a time
    # to prevent underflow issues
    for i, (xi, yi) in enumerate(zip(X, y)):
        # Get the mean of each component
        mu = coefficients.dot(xi)

        # Get the standard deviation of each component
        sigma = np.sqrt(variances)

        # Calculate the likelihood of this data point coming from each component
        temp_weights = norm.pdf(yi, loc=mu, scale=sigma)

        # Update the likelihood of each component generating this set
        assignment_weights[i] *= temp_weights / temp_weights.sum()
        assignment_weights[i] /= assignment_weights[i].sum()

        # Multiply in the component weightings
        assignment_weights[i] *= component_weights
        assignment_weights[i] /= assignment_weights[i].sum()

    return assignment_weights


def maximum_likelihood_parameters(X, y, num_components, assignments, assignment_weights):
    """
    Calculate the parameter values that maximize the likelihood of the data.
    This is the M-step of the CEM algorithm.

    Args:
        X (np.ndarray): source data (N x M matrix)
        y (np.ndarray): target data (N x K matrix)
        num_components (int): number of components to fit (J)
        assignments (np.ndarray): component assignment for all x,y pairs
        assignment_weights (np.ndarray): posterior prob for assignments

    Returns:
        component_weights (np.ndarray): prior probability (pi_j) (J x 1)
        coefficients (np.ndarray): coefficients for mean in linear regression, i.e., beta (J x M)
        variances (np.ndarray): variances associated with components (J x 1)
    """
    num_features = X.shape[1]

    # Calculate the weight of each component in the mixture (w_ij in [2])
    component_weights = np.array([(assignments == i).sum() for i in range(num_components)]) / float(len(assignments))  # J x 1

    # Calculate the regression coefficients and variance for each component
    coefficients = np.zeros((num_components, num_features))  # J x M
    variances = np.zeros(num_components)  #  J x 1
    for i in range(num_components):
        # Get the points that are members of this component
        points = np.where(assignments == i)[0]

        # Get the weights for each set
        weights = assignment_weights[points, i]

        # If no points were assigned to this cluster, soft-assign it random points
        # TODO: Is there a better way to proceed here? Some sort of split-merge type thing?
        if len(points) == 0:
            points = np.random.choice(len(assignments), size=np.random.randint(1, len(assignments)), replace=False)
            weights = np.ones(len(points)) / float(len(points))

        Xi, yi = X[points], y[points]

        # Get the weighted least squares coefficients
        coefficients[i] = weighted_linear_regression(Xi, yi, weights)

        # Get the variance of the component given the coefficients
        variances[i] = weighted_regression_variance(Xi, yi, weights, coefficients[i])

    return (component_weights, coefficients, variances)


def data_log_likelihood(X, y, assignments, component_weights, coefficients, variances):
    """
    Calculate the log-likelihood of the data being generated by the mixture model
    with the given parameters.

    Args:
        X (np.ndarray): source data (N x M matrix)
        y (np.ndarray): target data (N x 1 matrix)
        assignments (np.ndarray): component assignment for all x,y pairs
        component_weights (np.ndarray): prior probability (pi_j) (J x 1)
        coefficients (np.ndarray): coefficients for mean in linear regression, i.e., beta (J x M)
        variances (np.ndarray): variances associated with components (J x 1)

    Returns:
        log_likelihood (float): log likelihood of the data
    """

    log_likelihood = 0
    # TODO: speed this up
    for i in range(X.shape[0]):
        assigned = assignments[i]

        mu = X[i].dot(coefficients[assigned])
        sigma = np.sqrt(variances[assigned])

        log_likelihood += np.log(norm.pdf(y[i], loc=mu, scale=sigma)).sum()
        log_likelihood += np.log(component_weights[assigned])

    return log_likelihood


def fit_mixture(X, y, num_components, max_iterations, stochastic=False, threshold=1e-5):
    """
    Run the classification expecatation-maximization (CEM) algorithm to fit a maximum likelihood model.
    Note that the result is a local optimum, not necessarily a global one.

    Args:
        X (np.ndarray): source data (N x M matrix)
        y (np.ndarray): target data (N x 1 matrix)
        num_components (int): number of components to fit (J)
        max_iterations (int): maximum number of iterations to allow
        stochastic (bool): calculate weights with a stochastic algorithm or not
        threshold (float): threshold at which iterations stop

    Returns:
        results (MixtureResults): instance of the MixtureResults class which
            holds the determined coefficients and all that good stuff
    """
    # Initialize the results
    results = MixtureResults(num_components)

    prev_log_likelihood = 1
    cur_log_likelihood = 0
    cur_iteration = 0

    logger.info('Randomly initializing assignment weights')

    # Random initialization
    assignment_weights = np.random.uniform(size=(X.shape[0], num_components))  # N x J
    assignment_weights /= assignment_weights.sum(axis=1)[:, np.newaxis]  # (N x J) / (N x 1)

    logger.info('Sampling assignments')

    # Initialize using the normal steps now
    assignments = calculate_assignments(assignment_weights, stochastic=True)  # N x 1

    component_weights, coefficients, variances = maximum_likelihood_parameters(X, y, num_components,
                                                                               assignments, assignment_weights)

    while np.abs(prev_log_likelihood - cur_log_likelihood) > threshold and cur_iteration < max_iterations:
        logger.info('Starting iteration #{0}'.format(cur_iteration+1))

        # Calculate the expectation weights
        assignment_weights = calculate_assignment_weights(X, y, component_weights, coefficients, variances)  # N x J

        # Assign a value to each of the points
        assignments = calculate_assignments(assignment_weights, stochastic=stochastic)  # N x 1

        # Maximize the likelihood of the parameters  (J x 1, J x M, J x 1)
        component_weights, coefficients, variances = maximum_likelihood_parameters(X, y, num_components,
                                                                                   assignments, assignment_weights)

        # Calculate the total data log-likelihood
        prev_log_likelihood = cur_log_likelihood
        cur_log_likelihood = data_log_likelihood(X, y, assignments, component_weights, coefficients, variances)

        logger.info('Log-Likelihood: {0}'.format(cur_log_likelihood))

        # Add the iteration to the results
        results.add_iteration(assignments, component_weights, coefficients, variances, cur_log_likelihood)

        cur_iteration += 1

    # Tell the results that we're done fitting
    results.finish()

    return results


def __fit_worker(worker_params):
    X, y, num_components, max_iterations, stochastic = worker_params
    return fit_mixture(X, y, num_components, max_iterations, stochastic=stochastic)


def fit_with_restarts(X, y, num_components, max_iterations, num_restarts, stochastic=False, num_workers=1):
    """
    Run the CEM algorithm for num_restarts times and return the best result.

    Args:
        X (np.ndarray): source data (N x M matrix)
        y (np.ndarray): target data (N x 1 matrix)
        num_components (int): number of components to fit (J)
        max_iterations (int): maximum number of iterations to allow
        num_restarts (int): number of times to restart to find better local optimum
        stochastic (bool): calculate weights with a stochastic algorithm or not
        num_workers (int): number of processors to use for parallel processing

    Returns:
        max_result (MixtureResults): instance of the MixtureResults class which
            holds the determined coefficients and all that good stuff
    """
    max_result = None
    max_likelihood = None

    # Fit the mixture with every restart done in parallel
    if num_workers > 1 or num_workers == -1:
        pool = Pool(num_workers if num_workers > 1 else None)
        worker_params = [(X, y, num_components, max_iterations, stochastic) for _ in range(num_restarts)]
        results = pool.map(__fit_worker, worker_params)
    else:
        results = [fit_mixture(X, y, num_components, max_iterations, stochastic=stochastic) for _ in range(num_restarts)]

    for trial in range(num_restarts):
        result = results[trial]

        if max_likelihood is None or result.log_likelihoods.max() > max_likelihood:
            max_result = result
            max_likelihood = result.log_likelihoods.max()

    if num_workers > 1 or num_workers == -1:
        pool.terminate()

    return max_result


def __test():
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger()
    import matplotlib.pyplot as plt
    print('Testing mixture of 3 linear regression models with known number of components')

    NUM_COMPONENTS = 3
    NUM_POINTS = 1000
    TRUE_COEFFICIENTS = np.array([[2, 0.2], [1, 0.4], [0.8, -0.1]])
    TRUE_VARIANCES = np.array([0.1, 0.2, 0.1])**2
    TRUE_COMPONENT_WEIGHTS = np.array([0.4, 0.2, 0.4])
    TRUE_ASSIGNMENTS = np.random.choice(NUM_COMPONENTS, p=TRUE_COMPONENT_WEIGHTS, size=NUM_POINTS)

    # Generate some random points
    X = np.random.uniform(size=(NUM_POINTS, TRUE_COEFFICIENTS.shape[1]))
    X[:,0] = 1
    y = np.zeros((NUM_POINTS))

    for i in range(NUM_COMPONENTS):
        # Get the parameters of this subset's component
        coefficients = TRUE_COEFFICIENTS[i]
        variance = TRUE_VARIANCES[i]

        # Get class labels
        assignments = TRUE_ASSIGNMENTS == i

        # Generate a noisy version of the response variables
        y[assignments] = X[assignments].dot(coefficients) + np.random.normal(0, np.sqrt(variance), size=len(X[assignments]))

    results = fit_with_restarts(X, y, 3, 20, 20, stochastic=False, num_workers=1)

    # Plot the data
    COMPONENT_COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'brown', 'gray']
    for i in range(NUM_COMPONENTS):
        assignments = TRUE_ASSIGNMENTS == i
        Xi = X[assignments]
        yi = y[assignments]

        # Draw the data points
        plt.scatter(Xi[:,1], yi, color=COMPONENT_COLORS[i])

    # Plot the true lines
    for i,coefficients in enumerate(TRUE_COEFFICIENTS):
        x = np.linspace(0, 1, 6)
        features = np.ones((6, 2))
        features[:,1] = x
        y = features.dot(coefficients)
        plt.plot(x, y, color='gray', linestyle='--')

    # Plot the resulting lines
    for i, coefficients in enumerate(results.best.coefficients):
        x = np.linspace(0, 1, 6)
        features = np.ones((6, 2))
        features[:,1] = x
        y = features.dot(coefficients)

        plt.plot(x, y, color='black')

    plt.xlim(0,1)
    plt.show()


if __name__ == '__main__':
    __test()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.util.quality

measure synthesis image quality for the synthit package

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 26, 2018
"""

__all__ = ['synth_quality',
           'quality_simplex_area',
           'normalized_cross_correlation',
           'entropy_normalized_mutual_info',
           'mutual_info',
           'mssim']

import logging

import numpy as np
from skimage.measure import compare_ssim
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


def synth_quality(synth, truth, mask=None):
    """
    compare a synthesized image to the truth image by calculating metrics
    associated with image quality, the metrics are:
    (entropy normalized) mutual information, global correlation [2], and MSSIM [3]

    Args:
        synth (np.ndarray): synthesized image
        truth (np.ndarray): image we are trying to synthesize
        mask (np.ndarray): mask for the images

    Returns:
        stats (list): list of stats calculated from the corresponding metrics
        metrics (list): list of strings which define the calculated metrics

    References:
        [1] https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html
        [2] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
            “Image quality assessment: From error visibility to structural similarity,”
            IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.
    """
    metrics = ['ENMI', 'NCC', 'MSSIM']
    if mask is None:
        mask = truth > 0
    nmi = entropy_normalized_mutual_info(synth, truth, mask)
    gc = normalized_cross_correlation(synth, truth, mask)
    ssim = mssim(synth, truth, mask)
    stats = [nmi, gc, ssim]
    logger.info('ENMI: {:0.5f}, NCC: {:0.5f}, MSSIM: {:0.5f}'.format(nmi, gc, ssim))
    return stats, metrics


def quality_simplex_area(stats):
    """
    area of the "quality simplex," i.e., the area of the simplex defined in 2d by
    radially plotting the values of metrics mutual information, correlation, and ssim

    Args:
        stats (list): list of 3 statistics (whose entries are in the range [0,1])

    Returns:
        area (float): area of the "quality simplex" normalized between 0 and 1
    """
    stats_simplex = np.array([np.zeros(2) for _ in range(len(stats))])
    angles = np.linspace(0, 2 * np.pi, len(stats), endpoint=False)
    for i, (s, a) in enumerate(zip(stats, angles)):
        stats_simplex[i, :] = __pol2cart(s, a)
    # volume in 2d is area; denominator is normalization factor to force area to be in [0,1]
    area = ConvexHull(stats_simplex).volume  / (3 * np.sqrt(3) / 4)
    return area


def normalized_cross_correlation(x, y, mask=None):
    """ compute normalized cross correlation between two vectors/arrays """
    x = x if mask is None else x[mask == 1]
    y = y if mask is None else y[mask == 1]
    xm = x.mean()
    ym = y.mean()
    xhat = (x - xm) / np.sqrt(np.sum((x - xm) ** 2))
    yhat = (y - ym) / np.sqrt(np.sum((y - ym) ** 2))
    ncc = np.sum(xhat * yhat)
    return ncc


def entropy_normalized_mutual_info(x, y, mask=None, bins=100):
    """
    compute an entropy normalized mutual information (i.e., mutual information divided
    by its maximum value, specifically, the entropy of y)

    Args:
        x (np.ndarray): data, usually MRI image data
        y (np.ndarray): data, usually MRI image data
        mask (np.ndarray): mask of relevant data, usually a brain mask
        bins (int): number of bins to use in joint histogram on each axis

    Returns:
        nmi (float): normalized mutual information for x and y

    References:
        https://matthew-brett.github.io/teaching/mutual_information.html
    """
    if x.size != y.size:
        raise ValueError('input arrays must be equal for a valid mutual info')
    x_ = x.flatten() if mask is None else x[mask == 1]
    y_ = y.flatten() if mask is None else y[mask == 1]
    mi, pxy = mutual_info(x_, y_, bins, True)
    py = np.sum(pxy, axis=0)
    nzy = py > 0
    hy = -np.sum(py[nzy] * np.log2(py[nzy]))  # entropy of y
    nmi = mi / hy
    return nmi


def mutual_info(x, y, bins=200, return_joint=False):
    """ calculate the mutual information for two data arrays """
    joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = joint_hist / x.size
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals (now 2d)
    nzxy = pxy > 0
    mi = np.sum(pxy[nzxy] * np.log2(pxy[nzxy] / px_py[nzxy]))
    return mi, pxy if return_joint else mi


def mssim(x, y, mask=None):
    """ mean structural similarity (over a mask) """
    min_val = min(0, np.min(x), np.min(y))  # for some reason, calculations change when values are negative
    mssim, S = compare_ssim(x+min_val, y+min_val, full=True)
    if mask is not None:
        mssim = S[mask == 1].mean()
    return mssim


def __pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.util.io

measure synthesis image quality for the synthit package

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 26, 2018
"""

__all__ = ['synth_quality']

import logging

import ants
from skimage.measure import compare_ssim

logger = logging.getLogger(__name__)


def synth_quality(synth, truth, mask=None):
    """
    compare a synthesized image to the truth image by calculating metrics
    associated with image quality, the metrics are:
    mutual information [1], global correlation [2], and SSIM [3]

    Args:
        synth (ants.core.ants_image.ANTsImage): synthesized image
        truth (ants.core.ants_image.ANTsImage): image we are trying to synthesize
        mask (ants.core.ants_image.ANTsImage): mask for the images

    Returns:
        stats (list): list of stats calculated from the corresponding metrics
        metrics (list): list of strings which define the calculated metrics

    References:
        [1] D. Mattes, D. R. Haynor, H. Vesselle, T. K. Lewellyn, and W. Eubank,
            “Nonrigid multimodality image registration,” SPIE Med. Imaging, no. July 2001,
             pp. 1609–1620, 2001.
        [2] https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html
        [3] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli,
            “Image quality assessment: From error visibility to structural similarity,”
            IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, 2004.
    """
    metrics = ['MattesMutualInformation', 'Correlation', 'SSIM']
    stats = []
    for i, m in enumerate(metrics):
        logger.info('Calculating {} ({:d}/{:d})'.format(m, i+1, len(metrics)))
        if m != 'SSIM':
            stats.append(ants.image_similarity(truth, synth, m, mask, mask))
        else:
            _, S = compare_ssim(truth.numpy(), synth.numpy(), full=True)
            mssim = S[mask.numpy() == 1].mean()
            stats.append(mssim)
        logger.info('{}: {:0.5f}'.format(m, stats[i]))
    return stats, metrics

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.plot.synth_quality

create plot showing synthesis image quality

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 26, 2018
"""

__all__ = ['plot_dir_synth_quality',
           'plot_synth_quality']

import logging
import os

import ants
import matplotlib.pyplot as plt
import numpy as np

from ..errors import SynthError
from ..util.io import glob_nii, split_filename
from ..util.quality import synth_quality, quality_simplex

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    sns.set(style='whitegrid', font_scale=2, rc={'grid.color': '.9'})
except ImportError:
    logger.info('Seaborn not installed, so plots will not be as pretty. :-(')


def plot_dir_synth_quality(synth_dir, truth_dir, out_dir=None, mask_dir=None, outtype='png'):
    """
    compare directory of synthesized and truth (nifti) images by calculating
    correlation, mattes mutual information, and ssim and plotting a radar plot
    of the absolute value of those statistics

    Args:
        synth_dir (str): path to synthesized image directory
        truth_dir (str): path to truth image directory
        out_dir (str): path to save directory
        mask_dir (str): path to directory of corresponding masks (not needed)
        outtype (str): type of file to output (e.g., png, pdf, etc.)

    Returns:
        None
    """
    synth_fns = glob_nii(synth_dir)
    truth_fns = glob_nii(truth_dir)
    if len(synth_fns) != len(truth_fns) or len(synth_fns) == 0:
        raise SynthError('Number of synthesized and truth images must be equal and non-zero')
    if mask_dir is None:
        mask_fns = [None] * len(synth_fns)
    else:
        mask_fns = glob_nii(mask_dir)
        if len(synth_fns) != len(mask_fns):
            raise SynthError('Number of images and masks must be equal and non-zero')
    if out_dir is None:
        out_dir, _, _ = split_filename(synth_fns[0])
    for i, (synth_fn, truth_fn, mask_fn) in enumerate(zip(synth_fns, truth_fns, mask_fns), 1):
        _, base, _ = split_filename(synth_fn)
        logger.info('Comparing image: {} ({:d}/{:d})'.format(base, i, len(synth_fns)))
        mask = None if mask_fn is None else ants.image_read(mask_fn)
        synth = ants.image_read(synth_fn) if mask is None else ants.image_read(synth_fn) * mask
        truth = ants.image_read(truth_fn) if mask is None else ants.image_read(truth_fn) * mask
        _ = plot_synth_quality(synth, truth, mask)
        out_fn = os.path.join(out_dir, base + '.' + outtype)
        plt.savefig(out_fn)


def plot_synth_quality(synth, truth, mask):
    """
    create a radar plot of the (absolute value of the) metrics
    correlation, mattes mutual information, and ssim

    Args:
        synth (ants.core.ants_image.ANTsImage): synthesized image
        truth (ants.core.ants_image.ANTsImage): image we are trying to synthesize
        mask (ants.core.ants_image.ANTsImage): mask for the images

    Returns:
        ax (matplotlib ax object): ax that the plot was created on
    """
    stats, metrics = synth_quality(synth, truth, mask)
    stats = [np.abs(s) for s in stats]
    metrics = [m if m != 'MattesMutualInformation' else 'MI' for m in metrics]
    metrics = [m if m != 'Correlation' else 'GC' for m in metrics]  # that is, "global correlation"
    ax = __radar_plot(metrics, stats)
    area = quality_simplex(stats)
    ax.set_title('Synthesis Quality Simplex')
    ax.text(0.1, -0.1, 'Normalized\nSimplex Area: {:0.2f}'.format(area),
            transform=ax.transAxes, horizontalalignment='center')
    return ax


def __radar_plot(labels, stats):
    """
    create a radar plot from a list of labels and their corresponding stats

    References:
        https://www.kaggle.com/typewind/draw-a-radar-chart-with-python-in-a-simple-way
    """
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.grid(True)
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    ax.set_rmax(1)
    return ax

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
from ..util.quality import synth_quality, quality_simplex_area

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    sns.set(style='whitegrid', font_scale=2, rc={'grid.color': '.9'})
except ImportError:
    logger.info('Seaborn not installed, so plots will not be as pretty. :-(')


def plot_dir_synth_quality(synth_dir, truth_dir, out_dir=None, mask_dir=None, outtype='png', mean=False):
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
        mean (bool): option to use the mean of all images in directory

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
    if not mean:
        for i, (synth_fn, truth_fn, mask_fn) in enumerate(zip(synth_fns, truth_fns, mask_fns), 1):
            _, base, _ = split_filename(synth_fn)
            logger.info('Comparing image: {} ({:d}/{:d})'.format(base, i, len(synth_fns)))
            mask = None if mask_fn is None else ants.image_read(mask_fn)
            synth = ants.image_read(synth_fn) if mask is None else ants.image_read(synth_fn) * mask
            truth = ants.image_read(truth_fn) if mask is None else ants.image_read(truth_fn) * mask
            _ = plot_synth_quality(synth, truth, mask)
            out_fn = os.path.join(out_dir, base + '.' + outtype)
            plt.savefig(out_fn)
    else:
        masks = [None if mask_fn is None else ants.image_read(mask_fn) for mask_fn in mask_fns]
        synth = [ants.image_read(synth_fn) if mask is None else ants.image_read(synth_fn) * mask
                 for synth_fn, mask in zip(synth_fns, masks)]
        truth = [ants.image_read(truth_fn) if mask is None else ants.image_read(truth_fn) * mask
                 for truth_fn, mask in zip(truth_fns, masks)]
        _ = plot_synth_quality(synth, truth, masks, mean=True)
        out_fn = os.path.join(out_dir, 'mean_qs' + '.' + outtype)
        plt.savefig(out_fn)


def plot_synth_quality(synth, truth, mask, mean=False):
    """
    create a radar plot of the (absolute value of the) metrics
    correlation, mattes mutual information, and ssim

    Args:
        synth (ants.core.ants_image.ANTsImage or list): synthesized image
        truth (ants.core.ants_image.ANTsImage or list): image we are trying to synthesize
        mask (ants.core.ants_image.ANTsImage or list): mask for the images
        mean (bool): option to use the mean of all images in list
            (that is, synth, truth, and mask are lists of ANTsImages)

    Returns:
        ax (matplotlib ax object): ax that the plot was created on
    """
    if not mean:
        stats, metrics = synth_quality(synth.numpy(), truth.numpy(), mask.numpy())
        ax = __radar_plot(metrics, stats)
        area = quality_simplex_area(stats)
        title = 'Synthesis Quality Simplex'
    else:
        if not isinstance(synth, list) or not isinstance(truth, list) or not isinstance(mask, list):
            raise SynthError('If mean option used, then arguments must be lists (of equal size) '
                             'containing the images to be compared.')
        all_stats = np.zeros((len(synth), 3))
        for i, (synth_, truth_, mask_) in enumerate(zip(synth, truth, mask)):
            logger.info('Calculating image quality metrics ({:d}/{:d})'.format(i+1, len(synth)))
            stats_, metrics = synth_quality(synth_.numpy(), truth_.numpy(), mask_.numpy())
            all_stats[i, :] += np.array(stats_)
        mean_stats = all_stats.mean(0)
        std_stats = all_stats.std(0)
        ax = __radar_plot(metrics, mean_stats)
        ax = __radar_plot(metrics, mean_stats - std_stats, ax=ax, std=True)
        ax = __radar_plot(metrics, mean_stats + std_stats, ax=ax, std=True)
        area = quality_simplex_area(mean_stats)
        title = 'Mean Synthesis Quality Simplex'
    ax.set_title(title)
    ax.text(0.1, -0.1, 'Normalized\nSimplex Area: {:0.2f}'.format(area),
            transform=ax.transAxes, horizontalalignment='center')
    return ax


def __radar_plot(labels, stats, ax=None, std=False):
    """
    create a radar plot from a list of labels and their corresponding stats

    References:
        https://www.kaggle.com/typewind/draw-a-radar-chart-with-python-in-a-simple-way
    """
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    if not std:
        ax.plot(angles, stats, 'o-', linewidth=2)
        ax.fill(angles, stats, alpha=0.25)
        ax.grid(True)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        ax.set_rmax(1)
    else:
        ax.plot(angles, stats, 'k--', linewidth=1.5, alpha=0.25)
        ax.set_rmax(1)
    return ax

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.plot.directory_view

create profile view images of all images in a directory

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

__all__ = ['directory_view']

import logging
import os

import ants

from ..errors import SynthError
from ..util.io import glob_nii, split_filename

logger = logging.getLogger(__name__)


def directory_view(img_dir, out_dir=None, labels=None, figsize=3, outtype='png', slices=None, trim=True, scale=False):
    """
    create images for a directory of nifti files

    Args:
        img_dir (str): path to directory
        out_dir (str): path to save directory
        labels (str): path to directory of corresponding labels (not needed)
        figsize (float): size of output image
        outtype (str): type of file to output (e.g., png, pdf, etc.)
        slices (tuple): plot these slices in axial view (instead of ortho)
        trim (bool): trim blank/white space from image (need to have imagemagick installed)

    Returns:
        None
    """
    img_fns = glob_nii(img_dir)
    if labels is None:
        label_fns = [None] * len(img_fns)
    else:
        label_fns = glob_nii(labels)
        if len(img_fns) != len(label_fns):
            raise SynthError('Number of images and labels must be equal')
    if out_dir is None:
        out_dir, _, _ = split_filename(img_fns[0])
    for i, (img_fn, label_fn) in enumerate(zip(img_fns, label_fns), 1):
        _, base, _ = split_filename(img_fn)
        logger.info('Creating view for image: {} ({:d}/{:d})'.format(base, i, len(img_fns)))
        img = ants.image_read(img_fn)
        label = None if label_fn is None else ants.image_read(label_fn)
        out_fn = os.path.join(out_dir, base + '.' + outtype)
        if slices is None and hasattr(ants, 'plot_ortho'):
            ants.plot_ortho(img, overlay=label, overlay_cmap='prism', overlay_alpha=0.3,
                            flat=True, figsize=figsize, orient_labels=False, xyz_lines=False,
                            filename=out_fn)
        else:
            img = img.reorient_image2('ILP') if hasattr(img, 'reorient_image2') else img
            try:
                ants.plot(img, figsize=figsize, filename=out_fn, slices=slices,
                          reorient=False, scale=scale)
            except TypeError:
                import matplotlib.pyplot as plt
                ants.plot(img)
                plt.savefig(out_fn)
    if trim:
        logger.info('trimming blank space from all views')
        from subprocess import call
        try:
            call(['mogrify', '-trim', os.path.join(out_dir,'*.'+outtype)])
        except OSError as e:
            logger.warning('need to have imagemagick installed if trim option on')
            logger.error(e)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
directory_view

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


def directory_view(dir, out_dir=None, labels=None, figsize=3, outtype='png'):
    """
    create images for a directory of nifti files

    Args:
        dir (str): path to directory
        out_dir (str): path to save directory
        labels (str): path to directory of corresponding labels (not needed)
        figsize (float): size of output image
        outtype (str): type of file to output (e.g., png, pdf, etc.)

    Returns:
        None
    """
    img_fns = glob_nii(dir)
    if labels is not None:
        labels_fns = glob_nii(labels)
        if len(img_fns) != len(labels_fns):
            raise SynthError('Number of images and labels must be equal')
    if out_dir is None:
        out_dir, _, _ = split_filename(img_fns[0])
    for img_fn in img_fns:
        img = ants.image_read(img_fn)
        _, base, _ = split_filename(img_fn)
        out_fn = os.path.join(out_dir, base + '.' + outtype)
        ants.plot(img, figsize=figsize, filename=out_fn)

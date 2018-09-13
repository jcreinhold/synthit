#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.util.io

handle io operations for the synthit package

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

__all__ = ['split_filename',
           'glob_nii']

from glob import glob
import os


def split_filename(filepath):
    """ split a filepath into the directory, base, and extension """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def glob_nii(path):
    """ grab all nifti files in a directory and sort them for consistency """
    fns = sorted(glob(os.path.join(path, '*.nii*')))
    return fns


class AttrDict(dict):
    """
    make dictionary keys accessible via attributes
    used in nn_train and nn_predict to enable json config files
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

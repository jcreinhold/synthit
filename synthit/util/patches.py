#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.util.patches

handle the extraction of patches and reconstruction
from patches of 3d arrays (namely, 3d MR images)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

__all__ = ['extract_patches']

import numpy as np

from ..errors import SynthError


def extract_patches(data, idxs=None, patch_size=3, stride=1, min_val=0, ctx_radius=7):
    """
    extract patches (with or without context) from a 3D image

    Args:
        data (np.ndarray): 3d data
        idxs (tuple): tuple of np.ndarrays corresponding to indices (e.g., output from np.where)
        patch_size (int): patch size (this cubed), must be odd
        stride (int): use only (1/stide) of the indices
        context (bool): extract context for the patch or nah
        min_val (float): minimum value of extracted indices if idxs not provided
        ctx_radius (int): positive integer greater than patch size (0 if no context desired)

    Returns:
        patches (np.ndarray): array of patches
    """
    if idxs is None:
        idxs = np.where(data > min_val)
    if patch_size % 2 != 1:
        raise SynthError('Patch size must be odd')
    if len(idxs) != 3:
        raise SynthError('Data must be 3-dimensional.')
    context = True if ctx_radius > 0 else False
    idxs = [idx[::stride] for idx in idxs]
    patch_len = patch_size**3+6 if context else patch_size**3
    patches = np.zeros((len(idxs[0]), patch_len))
    h = int(np.floor(patch_size / 2))
    for n, (i, j, k) in enumerate(zip(*idxs)):
        #import ipdb; ipdb.set_trace()
        patch = data[i-h:i+h+1,j-h:j+h+1,k-h:k+h+1].flatten()
        ctx = patch_context(data, i, j, k, ctx_radius)
        patches[n, :] = np.concatenate((patch, ctx))
    return patches


def patch_context(data, i, j, k, r):
    idxs = (np.array([i+r,i-r,i,i,i,i]),
            np.array([j,j,j+r,j-r,j,j]),
            np.array([k,k,k,k,k+r,k-r]))
    ctx = data[idxs]
    return ctx

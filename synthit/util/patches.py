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


def extract_patches(data, idxs=None, patch_size=3, min_val=0, ctx_radius=(3,5,7), economy_patch=True):
    """
    extract patches (with or without context) from a 3D image

    Args:
        data (np.ndarray): 3d data
        idxs (tuple): tuple of np.ndarrays corresponding to indices (e.g., output from np.where)
        patch_size (int): patch size (this cubed), must be odd
        min_val (float): minimum value of extracted indices if idxs not provided
        ctx_radius (tuple): tuple of positive integers greater than patch size ((0) if no context desired)
        economy_patch (bool): return 'economy-sized' patches (not full patches, just the patch

    Returns:
        patches (np.ndarray): array of patches
    """
    if idxs is None:
        idxs = np.where(data > min_val)
    if patch_size % 2 != 1:
        raise SynthError('Patch size must be odd')
    if len(idxs) != 3:
        raise SynthError('Data must be 3-dimensional.')
    ctx_radius = ctx_radius if ctx_radius[0] > 0 else []
    if economy_patch:
        patch_len = 7 + len(ctx_radius) * 6
    else:
        patch_len = patch_size**3+(len(ctx_radius) * 6)
    patches = np.zeros((len(idxs[0]), patch_len))
    h = int(np.floor(patch_size / 2))
    for n, (i, j, k) in enumerate(zip(*idxs)):
        patch = get_patch(data, i, j, k, h, economy_patch).flatten()
        ctx = [patch_context(data, i, j, k, r) for r in ctx_radius]
        patches[n, :] = np.concatenate((patch, *ctx))
    return patches


def get_patch(data, i, j, k, h, economy_size=True):
    """
    get an individual patch based on the data and the indices and (half) patch size

    option to get an 'economy-sized' patch, which extracts the seemingly most important
    Args:
        data (np.ndarray): 3d data
        i (int): first coordinate index
        j (int): second coordinate index
        k (int): third coordinate index
        h (int): half-size of patch (if economy-sized, then this is not used)
        economy_size (bool): return 'economy-sized' patch or nah

    Returns:
        patch (np.ndarray): patch from data centered at i,j,k
    """
    if economy_size:
        # use patch_context for convenience, since it grabs desired elements
        patch = np.concatenate((np.array([data[i,j,k]]), patch_context(data, i, j, k, 1)))
    else:
        patch = data[i-h:i+h+1,j-h:j+h+1,k-h:k+h+1]
    return patch


def patch_context(data, i, j, k, r):
    """
    get context for a patch by extracting values outside of the patch
    in the three main directions

    Args:
        data (np.ndarray): 3d data
        i (int): first coordinate index
        j (int): second coordinate index
        k (int): third coordinate index
        r (int): radius (distance) to extract patches from

    Returns:
        ctx (np.ndarray): context values for a patch
    """
    idxs = (np.array([i+r,i-r,i,i,i,i]),
            np.array([j,j,j+r,j-r,j,j]),
            np.array([k,k,k,k,k+r,k-r]))
    ctx = data[idxs]
    return ctx

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.util.dataset

(pytorch) neural network nifti dataset handling tools

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 27, 2018
"""

__all__ = ['NiftiImageDataset',
           'RandomCrop']

from typing import Optional, Union, Callable

import ants
import numpy as np
import torch
from torch.utils.data import Dataset

from synthit.util.io import glob_nii


class NiftiImageDataset(Dataset):
    """
    create a nifti mri image (pytorch) dataset

    Args:
        source_dir (str): path to source images
        target_dir (str): path to target images
        crop (function or None): supply a cropping function
        plot (bool): output the image as antsimages for plotting
        disable_cuda (bool): disable CUDA even if it is available
    """

    def __init__(self, source_dir: str, target_dir: str,
                 crop: Optional[Callable]=None, plot: bool=False, disable_cuda: bool=False):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.source_fns = glob_nii(source_dir)
        self.target_fns = glob_nii(target_dir)
        if len(self.source_fns) != len(self.target_fns):
            raise ValueError('Number of source and target images must be equal')
        self.crop = crop
        self.plot = plot
        self.device = torch.device("cuda" if torch.cuda.is_available() and not disable_cuda else "cpu")

    def __len__(self):
        return len(self.source_fns)

    def __getitem__(self, idx: int):
        src_fn = self.source_fns[idx]
        tgt_fn = self.target_fns[idx]
        src_img = ants.image_read(src_fn)
        tgt_img = ants.image_read(tgt_fn)
        if src_img.spacing != tgt_img.spacing:
            raise ValueError('Resolution of source and target images need to be equal')
        if self.crop is not None:
            if self.plot:
                src_t, idxs = self.crop(src_img.numpy(), None)
                src_img = ants.from_numpy(src_t,
                                          src_img.origin,
                                          src_img.spacing,
                                          src_img.direction)
                tgt_t, _ = self.crop(tgt_img.numpy(), idxs)
                tgt_img = ants.from_numpy(tgt_t,
                                          src_img.origin,
                                          src_img.spacing,
                                          src_img.direction)
            else:
                src_t, idxs = self.crop(src_img.numpy(), None)
                src_img = torch.from_numpy(src_t)
                tgt_t, _ = self.crop(tgt_img.numpy(), idxs)
                tgt_img = torch.from_numpy(tgt_t)
        elif self.crop is None and not self.plot:
            # set to float32 to conserve memory
            src_img = torch.from_numpy(src_img.numpy().view(np.float32))
            tgt_img = torch.from_numpy(tgt_img.numpy().view(np.float32))
        if not self.plot:
            src_img = src_img.to(self.device)
            tgt_img = tgt_img.to(self.device)
        return src_img, tgt_img


class RandomCrop:
    """
    Randomly crop a 3d patch from a 3d image

    Args:
        output_size (tuple or int): Desired output size.
            If int, cube crop is made.
    """

    def __init__(self, output_size: Union[tuple, int]):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, img: np.ndarray, idxs: Optional[np.ndarray]=None):
        h, w, d = img.shape
        new_h, new_w, new_d = self.output_size

        if idxs is None:
            max_idxs = (h - new_h, w - new_w, d - new_d)
            mask = np.where(img > 0)
            c = np.random.randint(0, len(mask[0]))
            hh, ww, dd = [min(max_idxs[i], mask[i][c]) for i, m in enumerate(mask)]
        else:
            hh, ww, dd = idxs

        img = img[hh: hh + new_h,
              ww: ww + new_w,
              dd: dd + new_d]

        return img, (hh, ww, dd)

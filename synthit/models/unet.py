#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.models.unet

holds the architecture for a 3d unet [1]

References:
    [1] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
        “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
        in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 27, 2018
"""

import logging

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class Unet(torch.nn.Module):
    """
    defines a 3d unet [1] in pytorch

    Args:
        n_layers (int): number of layers (to go down and up)
        kernel_sz (int): size of kernel (symmetric)
        dropout_p (int): dropout probability for each layer

    References:
    [1] O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
        “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,”
        in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2016, pp. 424–432.

    """
    def __init__(self, n_layers: int, kernel_sz: int=3, dropout_p: float=0, patch_sz: int=64):
        super(Unet, self).__init__()
        self.n_layers = n_layers
        self.kernel_sz = kernel_sz
        self.dropout_p = dropout_p
        self.patch_sz = patch_sz
        def lc(n): return int(2 ** (5 + n))  # shortcut to layer count
        self.start = self.__dbl_conv_act(1, lc(0), lc(1))
        self.down_layers = nn.ModuleList([self.__dbl_conv_act(lc(n), lc(n), lc(n + 1))
                                          for n in range(1, n_layers)])
        self.bridge = self.__dbl_conv_act(lc(n_layers), lc(n_layers), lc(n_layers + 1))
        self.up_layers = nn.ModuleList([self.__dbl_conv_act(lc(n) + lc(n - 1), lc(n - 1), lc(n - 1))
                                        for n in reversed(range(3, n_layers + 2))])
        self.up_conv = nn.ModuleList([self.__conv(lc(n), lc(n))
                                      for n in reversed(range(2, n_layers + 2))])
        self.finish = self.__dbl_conv_act(lc(2) + lc(1), lc(1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.start(x)
        dout = []
        for dl in self.down_layers:
            dout.append(dl(x))
            x = F.max_pool3d(dout[-1], (2, 2, 2))
        x = self.up_conv[0](F.interpolate(self.bridge(x), scale_factor=2))
        for i, (ul, d) in enumerate(zip(self.up_layers, reversed(dout)), 1):
            x = ul(torch.cat((x, d), dim=1))
            x = self.up_conv[i](F.interpolate(x, scale_factor=2))
        x = self.finish(torch.cat((x, dout[0]), dim=1))
        return x

    def __conv(self, in_c: int, out_c: int) -> nn.Sequential:
        c = nn.Sequential(
            nn.ReplicationPad3d(self.kernel_sz // 2),
            nn.Conv3d(in_c, out_c, self.kernel_sz))
        return c

    def __conv_act(self, in_c: int, out_c: int) -> nn.Sequential:
        ca = nn.Sequential(
            self.__conv(in_c, out_c),
            nn.ReLU(),
            nn.InstanceNorm3d(out_c, affine=True),
            nn.Dropout3d(self.dropout_p))
        return ca

    def __dbl_conv_act(self, in_c: int, mid_c: int, out_c: int) -> nn.Sequential:
        dca = nn.Sequential(
            self.__conv_act(in_c, mid_c),
            self.__conv_act(mid_c, out_c))
        return dca

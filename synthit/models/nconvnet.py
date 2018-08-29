#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.models.nconvnet

define the class for a N layer CNN with
no max pool, increase in channels, or any of that
fancy stuff

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 27, 2018
"""

__all__ = ['Conv3dNLayerNet']

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class Conv3dNLayerNet(torch.nn.Module):
    def __init__(self, n_layers: int, n_channels: int=1, kernel_sz: int=3, dropout_p: float=0, patch_sz: int=64):
        super(Conv3dNLayerNet, self).__init__()
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.kernel_sz = kernel_sz
        self.dropout_p = dropout_p
        self.patch_sz = patch_sz
        if isinstance(kernel_sz, int):
            self.kernel_sz = [kernel_sz for _ in range(n_layers)]
        else:
            self.kernel_sz = kernel_sz
        self.layers = nn.ModuleList([nn.Sequential(
            nn.ReplicationPad3d(ksz//2),
            nn.Conv3d(n_channels, n_channels, ksz),
            nn.ReLU(),
            nn.InstanceNorm3d(n_channels, affine=True),
            nn.Dropout3d(dropout_p)) for ksz in self.kernel_sz])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x




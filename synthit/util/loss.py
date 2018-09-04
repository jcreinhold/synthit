#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.util.loss

custom pytorch loss functions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 01, 2018
"""

from typing import Callable

import torch


def histogram_loss(source: torch.Tensor, target: torch.Tensor, loss_func: Callable=torch.nn.MSELoss()):
    """
    compare the histograms of two tensors via `loss_func`

    Args:
        source (torch.Tensor): source tensor
        target (torch.Tensor): target tensor
        loss_func (Callable): loss function [default=torch.nn.MSELoss()]

    Returns:
        loss: value of loss_func given the hist of source and target
    """
    source_d, target_d = source.detach(), target.detach()
    min_val = min(torch.min(source_d), torch.min(target_d))
    max_val = max(torch.max(source_d), torch.max(target_d))
    tgt_pred_hist = torch.histc(target_d, min=min_val, max=max_val)
    tgt_hist = torch.histc(source_d, min=min_val, max=max_val)
    loss = loss_func(tgt_pred_hist, tgt_hist)
    return loss

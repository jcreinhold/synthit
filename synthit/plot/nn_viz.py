#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.plot.nn_viz

neural network visualization tools

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 28, 2018
"""

__all__ = ['plot_loss']

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    sns.set(style='white', font_scale=2)
except ImportError:
    logger.info('Seaborn not installed, so plots will not be as pretty. :-(')


def plot_loss(all_losses: List[list], figsize: Tuple[int,int]=(14,7), scale: int=-6, filename: Optional[str]=None):
    """
    plot loss vs epoch for a given list (of lists) of loss values

    Args:
        all_losses (list): list of lists of loss values per epoch
        figsize (tuple): two ints in a tuple controlling figure size
        scale (int): two ints in a tuple controlling figure size
        filename (str): if provided, save file at this path

    Returns:
        ax (matplotlib ax object): ax that the plot was created on
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    avg_losses = np.array([np.mean(losses) for losses in all_losses]) * (10 ** scale)
    std_losses = np.array([np.std(losses) for losses in all_losses]) * (10 ** scale)
    ax.errorbar(np.arange(len(avg_losses)), avg_losses, yerr=std_losses, ecolor='red')
    ax.set_title('Loss vs Epoch')
    ax.set_ylabel(r'MSE ($\times$' + f'10^{scale})')
    ax.set_xlabel('Epoch')
    ax.set_xlim((0, len(avg_losses)))
    ax.set_ylim((0, np.max(avg_losses) + 1))
    plt.savefig(filename)
    return ax

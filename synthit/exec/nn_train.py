#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.synth_train

command line interface to train a synthesis regressor

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 28, 2018
"""

import argparse
import logging
import sys
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from synthit import NiftiImageDataset, RandomCrop, SynthError


def arg_parser():
    parser = argparse.ArgumentParser(description='train a CNN for MR image synthesis')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--source-dir', type=str, required=True,
                          help='path to directory with source images')
    required.add_argument('-t', '--target-dir', type=str, required=True,
                          help='path to directory with target images')

    options = parser.add_argument_group('Options')
    options.add_argument('-o', '--output', type=str, default=None,
                         help='path to output the trained model')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                         help='optional directory of brain masks for images')
    options.add_argument('-na', '--nn-arch', type=str, default='unet', choices=('unet', 'nconv'),
                         help='specify neural network architecture to use')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")

    synth_options = parser.add_argument_group('Synthesis Options')
    synth_options.add_argument('--patch-size', type=int, default=64,
                               help='patch size^3 extracted from image [Default=64]')

    regr_options = parser.add_argument_group('Neural Network Options')
    regr_options.add_argument('-n', '--n-jobs', type=int, default=4,
                              help='number of processors to use [Default=4]')
    regr_options.add_argument('-ne', '--n-epochs', type=int, default=100,
                              help='number of epochs [Default=100]')
    regr_options.add_argument('-nl', '--n-layers', type=int, default=3,
                              help='number of layers to use in network (different meaning per arch) [Default=3]')
    regr_options.add_argument('-ks', '--kernel-size', type=int, default=3,
                              help='convolutional kernel size (cubed) [Default=3]')
    regr_options.add_argument('-dp', '--dropout-prob', type=float, default=0,
                              help='dropout probability per conv block [Default=0]')
    regr_options.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                              help='learning rate of the neural network (uses Adam) [Default=1e-3]')
    regr_options.add_argument('-bs', '--batch-size', type=int, default=5,
                              help='batch size (num of images to process at once) [Default=5]')
    regr_options.add_argument('--plot-loss', type=str, default=None,
                              help='plot the loss vs epoch and save at the filename provided here [Default=None]')
    regr_options.add_argument('--random-seed', default=0,
                              help='set random seed for reproducibility [Default=0]')
    return parser


def main():
    args = arg_parser().parse_args()
    if args.verbosity == 1:
        level = logging.getLevelName('INFO')
    elif args.verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger(__name__)
    try:
        np.random.seed(args.random_seed)
        if args.nn_arch == 'nconv':
            from synthit.models.nconvnet import Conv3dNLayerNet
            model = Conv3dNLayerNet(args.n_layers, kernel_sz=args.kernel_size, dropout_p=args.dropout_prob)
        elif args.nn_arch == 'unet':
            from synthit.models.unet import Unet
            model = Unet(args.n_layers, kernel_sz=args.kernel_size, dropout_p=args.dropout_prob)
        else:
            raise SynthError(f'Invalid NN type: {args.nn_arch}. {{nconv, unet}} are the only supported options.')
        logger.debug(model)

        # set up data loader for nifti images
        dataloader = DataLoader(NiftiImageDataset(args.source_dir, args.target_dir, crop=RandomCrop(args.patch_size)),
                                batch_size=args.batch_size, shuffle=True, num_workers=args.n_jobs)

        # train the model
        criterion = nn.MSELoss()
        logger.info(f'LR: {args.learning_rate:.5f}')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        all_losses = []
        for t in range(args.n_epochs):
            losses = []
            for batch in dataloader:
                x, y = batch
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = model(x.unsqueeze(1))  # add (empty) channel axis

                # Compute and print loss
                loss = criterion(y_pred, y.unsqueeze(1))
                logger.info(f'Epoch: {t+1}, Loss: {loss.item():.2f}')
                losses.append(loss.item())

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            all_losses.append(losses)

        # save the whole model (if changes occur to pytorch, then this model will probably not be loadable)
        torch.save(model, args.output)

        # plot the loss vs epoch (if desired)
        if args.plot_loss is not None:
            from synthit import plot_loss
            _ = plot_loss(all_losses, filename=args.plot_loss)

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

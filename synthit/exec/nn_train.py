#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.nn_train

command line interface to train a deep convolutional neural network for
synthesis of MR (brain) images

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 28, 2018
"""

import argparse
import logging
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import matplotlib
    matplotlib.use('agg')  # do not pull in GUI
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    from synthit import NiftiImageDataset, RandomCrop, SynthError
    from synthit.util.io import AttrDict


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
    options.add_argument('-vc', '--validation-count', type=int, default=0,
                         help="number of datasets to use in validation")
    options.add_argument('-ocf', '--out-config-file', type=str, default=None,
                         help='output a config file for the options used in this experiment '
                              '(saves them as a json file with the name as input in this argument)')

    synth_options = parser.add_argument_group('Synthesis Options')
    synth_options.add_argument('-ps', '--patch-size', type=int, default=64,
                               help='patch size^3 extracted from image [Default=64]')
    synth_options.add_argument('-cm', '--crop-mask', action='store_true', default=False,
                               help='crop the image according to the provided masks (or non-zero values). '
                                    'assumes skull-stripped images [Default=False]')

    nn_options = parser.add_argument_group('Neural Network Options')
    nn_options.add_argument('-n', '--n-jobs', type=int, default=0,
                            help='number of CPU processors to use (use 0 if CUDA enabled) [Default=0]')
    nn_options.add_argument('-ne', '--n-epochs', type=int, default=100,
                            help='number of epochs [Default=100]')
    nn_options.add_argument('-nl', '--n-layers', type=int, default=3,
                            help='number of layers to use in network (different meaning per arch) [Default=3]')
    nn_options.add_argument('-ks', '--kernel-size', type=int, default=3,
                            help='convolutional kernel size (cubed) [Default=3]')
    nn_options.add_argument('-dp', '--dropout-prob', type=float, default=0,
                            help='dropout probability per conv block [Default=0]')
    nn_options.add_argument('-lr', '--learning-rate', type=float, default=1e-3,
                            help='learning rate of the neural network (uses Adam) [Default=1e-3]')
    nn_options.add_argument('-bs', '--batch-size', type=int, default=5,
                            help='batch size (num of images to process at once) [Default=5]')
    nn_options.add_argument('-cbp', '--channel-base-power', type=int, default=5,
                            help='batch size (num of images to process at once) [Default=5]')
    nn_options.add_argument('--plot-loss', type=str, default=None,
                            help='plot the loss vs epoch and save at the filename provided here [Default=None]')
    nn_options.add_argument('--use-up-conv', action='store_true', default=False,
                            help='Use resize-convolution in the U-net as per the Distill article: '
                                 '"Deconvolution and Checkerboard Artifacts" [Default=False]')
    nn_options.add_argument('--add-two-up', action='store_true', default=False,
                            help='Add two to the kernel size on the upsampling in the U-Net as '
                                 'per Zhao, et al. 2017 [Default=False]')
    nn_options.add_argument('-nm', '--normalization', type=str, default='instance', choices=('instance', 'batch', 'none'),
                            help='type of normalization layer to use in network [Default=instance]')
    nn_options.add_argument('-ac', '--activation', type=str, default='relu', choices=('relu', 'lrelu'),
                            help='type of activation to use throughout network except output [Default=relu]')
    nn_options.add_argument('-oac', '--out-activation', type=str, default='linear', choices=('relu', 'lrelu', 'linear'),
                            help='type of activation to use in network on output [Default=linear]')
    nn_options.add_argument('--disable-cuda', action='store_true', default=False,
                            help='Disable CUDA regardless of availability')
    return parser


def main(args=None):
    no_config_file = args is not None or (args is None and len(sys.argv[1:]) > 1)
    if no_config_file:
        args = arg_parser().parse_args(args)
    else:
        import json
        with open(sys.argv[1:][0], 'r') as f:
            args = AttrDict(json.load(f))
    if args.verbosity == 1:
        level = logging.getLevelName('INFO')
    elif args.verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger(__name__)
    try:
        # set number of threads if using CPU
        torch.set_num_threads(args.n_jobs)

        # get the desired neural network architecture
        if args.nn_arch == 'nconv':
            from synthit.models.nconvnet import Conv3dNLayerNet
            logger.warning('The nconv network is for basic testing.')
            model = Conv3dNLayerNet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size)
        elif args.nn_arch == 'unet':
            from synthit.models.unet import Unet
            model = Unet(args.n_layers, kernel_size=args.kernel_size, dropout_p=args.dropout_prob, patch_size=args.patch_size,
                         channel_base_power=args.channel_base_power, add_two_up=args.add_two_up, normalization=args.normalization,
                         activation=args.activation, output_activation=args.out_activation, use_up_conv=args.use_up_conv)
        else:
            raise SynthError(f'Invalid NN type: {args.nn_arch}. {{nconv, unet}} are the only supported options.')
        logger.debug(model)

        # put the model on the GPU if available and desired
        if torch.cuda.is_available() and not args.disable_cuda:
            model.cuda()

        # control random cropping patch size (or if used at all)
        crop = RandomCrop(args.patch_size) if args.patch_size > 0 else None

        # if crop mask enabled do not use predefined crop and set batch size to 1, patch size set to zero for consistency
        if args.crop_mask:
            if args.batch_size > 1:
                logger.info('If crop-mask option enabled, then batch size is automatically set to 1.')
                args.batch_size = 1
            args.patch_size = 0
            crop = None

        # define dataset and split into training/validation set
        dataset = NiftiImageDataset(args.source_dir, args.target_dir, crop=crop, disable_cuda=args.disable_cuda)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = args.validation_count
        validation_idx = np.random.choice(indices, size=split, replace=False)
        train_idx = list(set(indices) - set(validation_idx))

        train_sampler = SubsetRandomSampler(train_idx)
        validation_sampler = SubsetRandomSampler(validation_idx)

        # set up data loader for nifti images
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.n_jobs)
        validation_loader = DataLoader(dataset, sampler=validation_sampler, batch_size=args.batch_size, num_workers=args.n_jobs)

        # train the model
        criterion = nn.MSELoss()
        logger.info(f'LR: {args.learning_rate:.5f}')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        train_losses, validation_losses = [], []
        for t in range(args.n_epochs):
            # training
            losses = []
            for src, tgt in train_loader:
                # Forward pass: Compute predicted y by passing x to the model
                tgt_pred = model(src.unsqueeze(1))  # add (empty) channel axis

                # Compute and print loss
                loss = criterion(tgt_pred, tgt.unsqueeze(1))
                logger.info(f'Training - Epoch: {t+1}, Loss: {loss.item():.2f}')
                losses.append(loss.item())

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_losses.append(losses)

            # validation
            losses = []
            for src, tgt in validation_loader:
                tgt_pred = model(src.unsqueeze(1))  # add (empty) channel axis

                # Compute and print loss
                loss = criterion(tgt_pred, tgt.unsqueeze(1))
                logger.info(f'Validation - Epoch: {t+1}, Loss: {loss.item():.2f}')
                losses.append(loss.item())

            validation_losses.append(losses)

        # output a config file if desired
        if args.out_config_file is not None:
            import json
            arg_dict = vars(args)
            # add these keys so that the output config file can be edited for use in prediction
            arg_dict['trained_model'] = args.output
            arg_dict['predict_dir'] = None
            arg_dict['predict_out'] = None
            arg_dict['predict_mask_dir'] = None
            with open(args.out_config_file, 'w') as f:
                json.dump(arg_dict, f, sort_keys=True, indent=2)

        # save the trained model
        if not no_config_file or args.out_config_file is not None:
            torch.save(model.state_dict(), args.output)
        else:
            # save the whole model (if changes occur to pytorch, then this model will probably not be loadable)
            logger.warning('Saving the entire model. Preferred to create a config file and only save model weights')
            torch.save(model, args.output)

        # plot the loss vs epoch (if desired)
        if args.plot_loss is not None:
            plot_error = True if args.n_epochs <= 50 else False
            from synthit import plot_loss
            if matplotlib.get_backend() != 'agg':
                import matplotlib.pyplot as plt
                plt.switch_backend('agg')
            ax = plot_loss(train_losses, ecolor='maroon', label='Train', plot_error=plot_error)
            _ = plot_loss(validation_losses, filename=args.plot_loss, ecolor='firebrick', ax=ax, label='Validation', plot_error=plot_error)

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

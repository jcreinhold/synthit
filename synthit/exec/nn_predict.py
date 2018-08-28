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
import torch

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import ants
    from synthit import glob_nii


def arg_parser():
    parser = argparse.ArgumentParser(description='train a CNN for MR image synthesis')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--source-dir', type=str, required=True,
                          help='path to directory with source images')
    required.add_argument('-t', '--trained-model', type=str, required=True,
                          help='path to trained model')

    options = parser.add_argument_group('Options')
    options.add_argument('-o', '--output', type=str, default=None,
                         help='path to output the synthesized image')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                         help='optional directory of brain masks for images')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")

    synth_options = parser.add_argument_group('Synthesis Options')
    synth_options.add_argument('--patch-size', type=int, default=64,
                               help='patch size^3 extracted from image [Default=64]')

    regr_options = parser.add_argument_group('Neural Network Options')
    regr_options.add_argument('-bs', '--batch-size', type=int, default=5,
                              help='batch size (num of images to process at once) [Default=5]')
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
        psz = args.patch_size
        model = torch.load(args.trained_model)
        logger.debug(model)

        source_fns = glob_nii(args.source_dir)
        for fn in source_fns:
            img_ants = ants.image_read(fn)
            img = img_ants.numpy()
            if psz > 0:
                out_img = np.zeros(img.shape)
                count_mtx = np.zeros(img.shape)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stride = psz // 2
                    indices = [torch.from_numpy(idxs) for idxs in np.indices(img.shape)]
                    for i in range(3):  # create blocks from imgs (and indices)
                        indices = [idxs.unfold(i, psz, stride) for idxs in indices]
                    x, y, z = [idxs.contiguous().view(-1, psz, psz, psz) for idxs in indices]
                dec_idxs = np.floor(np.percentile(np.arange(x.shape[0]), np.arange(0, 101, 5)))
                count = 0
                for i, (xx, yy, zz) in enumerate(zip(x, y, z)):
                    if i in dec_idxs:
                        logger.info(f'{count}% Complete')
                        count += 5
                    patch = torch.from_numpy(img[xx, yy, zz])[None, None, ...]
                    predicted = np.squeeze(model.forward(patch).data.numpy())
                    out_img[xx, yy, zz] = out_img[xx, yy, zz] + predicted
                    count_mtx[xx, yy, zz] = count_mtx[xx, yy, zz] + 1
                count_mtx[count_mtx == 0] = 1  # avoid division by zero
                out_img_ants = img_ants.new_image_like(out_img / count_mtx)
            else:
                test_img_t = torch.from_numpy(img)
                out_img = np.squeeze(model.forward(test_img_t).data.numpy())
                out_img_ants = img_ants.new_image_like(out_img)
            out_img_ants.to_filename(args.output)

        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

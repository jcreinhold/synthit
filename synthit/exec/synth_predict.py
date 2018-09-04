#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.synth_predict

command line interface to synthesize images with a patch-based trained regressor

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

import argparse
from glob import glob
import logging
import os
import sys
import warnings

import ants
from sklearn.externals import joblib

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from synthit import split_filename, glob_nii, SynthError


def arg_parser():
    parser = argparse.ArgumentParser(description='synthesize MR images via patch-based regression')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--source-dir', type=str, required=True, nargs='+',
                        help='path to directory with domain images')
    required.add_argument('-t', '--trained-model', type=str, required=True,
                          help='path to the trained model (.pkl)')

    options = parser.add_argument_group('Options')
    options.add_argument('-o', '--output-dir', type=str, default=None,
                         help='path to output the synthesized images')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                         help='optional directory of brain masks for images')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    options.add_argument('--cross-validate', action='store_true', default=False,
                         help='do leave one out cross-validation on the provided dataset (e.g., if 5 datasets are '
                              'provided, then 5 models are trained where all the data are used except one).')
    return parser


def process(ps, img_fn, mask_fn, k, n, logger, args):
    img_fn = img_fn[0]
    dirpath, base, _ = split_filename(img_fn[0])
    logger.info('Synthesizing image from: {} ({:d}/{:d})'.format(base, k, n))
    mask = None if mask_fn is None else ants.image_read(mask_fn)
    img = [ants.image_read(img_fn_) if mask is None else ants.image_read(img_fn_) * mask for img_fn_ in img_fn]
    synth = ps.predict(img, mask)
    out_fn = os.path.join(dirpath if args.output_dir is None else args.output_dir, base + '_syn.nii.gz')
    logger.info('Saving image: {}'.format(out_fn))
    synth.to_filename(out_fn)


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
        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
        img_fns = [glob_nii(sd) for sd in args.source_dir]
        mask_fns = [None] * len(img_fns[0]) if args.mask_dir is None else glob_nii(args.mask_dir)
        if not args.cross_validate:
            ps = joblib.load(args.trained_model)
            for i, (*img_fn, mask_fn) in enumerate(zip(zip(*img_fns), mask_fns), 1):
                process(ps, img_fn, mask_fn, i, len(mask_fns), logger, args)
        else:
            if os.path.isdir(args.trained_model):
                trained_models = sorted(glob(os.path.join(args.trained_model, '*.pkl')))
            else:
                raise SynthError('If cross-validate option used, then trained_model argument must be a '
                                 'directory filled with the trained models')
            if len(mask_fns) != len(trained_models):
                raise SynthError('Must use the entire set of training data when using the cross-validate option')
            for i, tm in enumerate(trained_models):
                ps = joblib.load(tm)
                img_fns_ = [[img_fn_ for k, img_fn_ in enumerate(img_fns_) if i == k] for img_fns_ in img_fns]
                mask_fns_ = [mask_fn_ for k, mask_fn_ in enumerate(mask_fns) if i == k]
                logger.info('Cross-validation for {} ({:d}/{:d})'.format(img_fns_[0][0], i+1, len(trained_models)))
                for k, (*img_fn, mask_fn) in enumerate(zip(zip(*img_fns_), mask_fns_), 1):
                    process(ps, img_fn, mask_fn, k, len(mask_fns_), logger, args)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.synth_predict

command line interface to synthesize images with a patch-based trained regressor

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

import argparse
import logging
import os
import sys
import warnings

import ants
from sklearn.externals import joblib

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from synthit import split_filename, glob_nii


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
        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
        ps = joblib.load(args.trained_model)
        img_fns = [glob_nii(sd) for sd in args.source_dir]
        mask_fns = [None] * len(img_fns[0]) if args.mask_dir is None else glob_nii(args.mask_dir)
        for i, (*img_fn, mask_fn) in enumerate(zip(zip(*img_fns), mask_fns), 1):
            img_fn = img_fn[0]
            dirpath, base, _ = split_filename(img_fn[0])
            logger.info('Synthesizing image from: {} ({:d}/{:d})'.format(base, i, len(mask_fns)))
            mask = None if mask_fn is None else ants.image_read(mask_fn)
            img = [ants.image_read(img_fn_) if mask is None else ants.image_read(img_fn_) * mask for img_fn_ in img_fn]
            synth = ps.predict(img, mask)
            out_fn = os.path.join(dirpath if args.output_dir is None else args.output_dir, base + '_syn.nii.gz')
            logger.info('Saving image: {}'.format(out_fn))
            synth.to_filename(out_fn)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

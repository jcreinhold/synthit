#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.synth_quality

command line interface to create synthesis quality radar plot for a directory of images

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

import argparse
import logging
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from synthit import plot_dir_synth_quality, plot_synth_quality_bar


def arg_parser():
    parser = argparse.ArgumentParser(description='create profile views of every nifti image in a directory')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--synth-dir', type=str, required=True, nargs='+',
                        help='path to directory of synthesized images')
    required.add_argument('-t', '--truth-dir', type=str, required=True,
                          help='path to corresponding truth images')

    options = parser.add_argument_group('Optional')
    required.add_argument('-na', '--norm-algs', type=str, nargs='+',
                          help='normalizaiton algorithms, must be provided if multiple directories provided')
    required.add_argument('-sa', '--synth-algs', type=str, nargs='+',
                          help='synthesis algorithms, must be provided if multiple directories provided')
    options.add_argument('-o', '--output-dir', type=str, default=None,
                         help='directory to output the corresponding views')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                         help='optional directory of labels for images')
    options.add_argument('-ot', '--output-type', type=str, default='png',
                         help='type of output image to save (e.g., png, pdf, etc.)')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    options.add_argument('--mean', action='store_true', default=False,
                         help="plot the mean of the quality metrics for the directory")
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
        if len(args.synth_dir) == 1:
            plot_dir_synth_quality(args.synth_dir[0], args.truth_dir[0], args.output_dir,
                                   args.mask_dir, args.output_type, args.mean)
        else:
            _ = plot_synth_quality_bar(args.synth_dir, args.truth_dir, args.norm_algs, args.synth_algs,
                                       args.output_dir, args.mask_dir, args.output_type)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

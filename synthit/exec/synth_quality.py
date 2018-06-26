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
    from synthit import plot_dir_synth_quality


def arg_parser():
    parser = argparse.ArgumentParser(description='create profile views of every nifti image in a directory')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--synth-dir', type=str, required=True,
                        help='path to directory of synthesized images')
    required.add_argument('-t', '--truth-dir', type=str, required=True,
                          help='path to corresponding truth images')

    options = parser.add_argument_group('Optional')
    options.add_argument('-o', '--output-dir', type=str, default=None,
                         help='directory to output the corresponding views')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                         help='optional directory of labels for images')
    options.add_argument('-ot', '--output-type', type=str, default='png',
                         help='type of output image to save (e.g., png, pdf, etc.)')
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
        plot_dir_synth_quality(args.synth_dir, args.truth_dir, args.output_dir, args.mask_dir, args.output_type)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

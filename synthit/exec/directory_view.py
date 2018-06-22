#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.directory_view

command line interface to create profile images for a directory of nifti files

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

import argparse
import logging
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from synthit import directory_view


def arg_parser():
    parser = argparse.ArgumentParser(description='create profile views of every nifti image in a directory')

    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed '
                             '(should all be T1w contrast)')

    options = parser.add_argument_group('Optional')
    options.add_argument('-o', '--output-dir', type=str, default=None,
                         help='directory to output the corresponding views')
    options.add_argument('-l', '--label-dir', type=str, default=None,
                         help='optional directory of labels for images')
    options.add_argument('-f', '--figsize', type=float, default=3,
                         help='size of output image')
    options.add_argument('-ot', '--output-type', type=str, default='png',
                         help='type of output image to save (e.g., png, pdf, etc.)')
    options.add_argument('--slices', action='store_true', default=False,
                         help='plot slices instead of ortho view')
    options.add_argument('--trim', action='store_true', default=False,
                         help='trim output image of blank/white space outside the plot')
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
        directory_view(args.img_dir, args.output_dir, args.label_dir, args.figsize,
                       args.output_type, not args.slices, args.trim)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

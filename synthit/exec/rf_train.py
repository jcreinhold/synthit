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
import pickle
import sys
import warnings

from sklearn.ensemble import RandomForestRegressor

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from synthit import PatchSynth


def arg_parser():
    parser = argparse.ArgumentParser(description='train a rf synthesis')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--source-dir', type=str, required=True,
                        help='path to directory with domain images')
    required.add_argument('-t', '--target-dir', type=str, required=True,
                          help='path to directory with target images')

    options = parser.add_argument_group('Options')
    options.add_argument('-o', '--output', type=str, default=None,
                         help='path to output the trained random forest')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                         help='optional directory of brain masks for images')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")

    adv_options = parser.add_argument_group('Advanced Options')
    adv_options.add_argument('-n', '--n-jobs', type=int, default=-1,
                             help='number of processors to use (-1 is all processors) '
                                  '[Default=-1]')
    adv_options.add_argument('--min-leaf', type=int, default=5,
                             help='minimum number of leaves (see min_samples_leaf) [Default=5]')
    adv_options.add_argument('--n-trees', type=int, default=60,
                             help='number of trees in rf (see n_estimators) [Default=60]')
    adv_options.add_argument('--max-features', default=(1.0/3.0),
                             help='proportion of features to use in rf (see max_features) [Default=1/3]')
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
        rf = RandomForestRegressor(n_jobs=args.n_jobs, min_samples_leaf=args.min_leaf,
                                   n_estimators=args.n_trees, max_features=args.max_features)
        ps = PatchSynth(rf, flatten=True)
        source = ps.image_list(args.source_dir)
        target = ps.image_list(args.target_dir)
        if args.mask_dir is not None:
            masks = ps.image_list(args.mask_dir)
            source = [src * mask for (src, mask) in zip(source, masks)]
            target = [tgt * mask for (tgt, mask) in zip(target, masks)]
        ps.fit(source, target)
        outfile = 'trained_rf.pkl' if args.output is None else args.output
        with open(outfile, 'wb') as f:
            pickle.dump(ps, f)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

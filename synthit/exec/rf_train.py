#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.rf_train

command line interface to train a synthesis random forest regressor

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

import argparse
import logging
import sys
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    from synthit import PatchSynth


def arg_parser():
    parser = argparse.ArgumentParser(description='train a patch-based RF regressor for MR image synthesis')

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

    synth_options = parser.add_argument_group('Synthesis Options')
    synth_options.add_argument('--patch-size', type=int, default=3,
                               help='patch size extracted for regression [Default=3]')
    synth_options.add_argument('--full-patch', action='store_true', default=False,
                               help='use the full patch in regression vs a reduced size patch [Default=True]')
    synth_options.add_argument('--stride', type=int, default=1,
                               help='use every `stride` voxel in regressor [Default=1]')
    synth_options.add_argument('--ctx-radius', type=int, default=(3,5), nargs='+',
                               help='context radii to use when extracting patches [Default=(3,5)]')
    synth_options.add_argument('--threshold', type=int, default=0,
                               help='threshold for foreground and background (above is foreground) [Default=0]')

    regr_options = parser.add_argument_group('Regressor Options')
    regr_options.add_argument('-n', '--n-jobs', type=int, default=-1,
                              help='number of processors to use (-1 is all processors) [Default=-1]')
    regr_options.add_argument('--min-leaf', type=int, default=5,
                              help='minimum number of leaves (see min_samples_leaf) [Default=5]')
    regr_options.add_argument('--n-trees', type=int, default=60,
                              help='number of trees in rf (see n_estimators) [Default=60]')
    regr_options.add_argument('--max-features', default=(1.0/3.0),
                              help='proportion of features to use in rf (see max_features) [Default=1/3]')
    regr_options.add_argument('--random-seed', default=0,
                              help='set random seed of the rf for reproducibility [Default=0]')
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
        regr = RandomForestRegressor(n_jobs=args.n_jobs, min_samples_leaf=args.min_leaf,
                                     n_estimators=args.n_trees, max_features=args.max_features,
                                     random_state=args.random_seed, verbose=1 if args.verbosity >= 2 else 0)
        logger.debug(regr)
        ps = PatchSynth(regr, args.patch_size, args.stride, args.ctx_radius, args.threshold, flatten=True)
        source = ps.image_list(args.source_dir)
        target = ps.image_list(args.target_dir)
        if args.mask_dir is not None:
            masks = ps.image_list(args.mask_dir)
            source = [src * mask for (src, mask) in zip(source, masks)]
            target = [tgt * mask for (tgt, mask) in zip(target, masks)]
        ps.fit(source, target)
        outfile = 'trained_rf.pkl' if args.output is None else args.output
        logger.info('Saving trained model: {}'.format(outfile))
        joblib.dump(ps, outfile)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

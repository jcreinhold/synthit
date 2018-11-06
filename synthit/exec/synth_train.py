#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
synthit.exec.synth_train

command line interface to train a synthesis regressor

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

import argparse
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import numpy as np
    from sklearn.externals import joblib
    from synthit import PatchSynth, SynthError


def arg_parser():
    parser = argparse.ArgumentParser(description='train a patch-based regressor for MR image synthesis')

    required = parser.add_argument_group('Required')
    required.add_argument('-s', '--source-dir', type=str, required=True, nargs='+',
                        help='path to directory with domain images '
                             '(multiple paths can be provided for multi-modal synthesis, '
                             'put T1-w images first if they are not skull-stripped)')
    required.add_argument('-t', '--target-dir', type=str, required=True,
                          help='path to directory with target images')

    options = parser.add_argument_group('Options')
    options.add_argument('-o', '--output', type=str, default=None,
                         help='path to output the trained regressor')
    options.add_argument('-m', '--mask-dir', type=str, default=None,
                         help='optional directory of brain masks for images')
    options.add_argument('-r', '--regr-type', type=str, default='rf', choices=('rf', 'xg', 'pr', 'mlr', 'mlp'),
                         help='specify type of regressor to use')
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    options.add_argument('--cross-validate', action='store_true', default=False,
                         help='do leave one out cross-validation on the provided dataset (e.g., if 5 datasets are '
                              'provided, then 5 models are trained where all the data are used except one).')

    synth_options = parser.add_argument_group('Synthesis Options')
    synth_options.add_argument('-ps', '--patch-size', type=int, default=3,
                               help='patch size extracted for regression [Default=3]')
    synth_options.add_argument('-fp', '--full-patch', action='store_true', default=False,
                               help='use the full patch in regression vs a reduced size patch [Default=False]')
    synth_options.add_argument('-ns', '--n-samples', type=float, default=None,
                               help='use randomly sampled (with replacement) `n_samples` voxels for training '
                                    'regressor (None uses all voxels) [Default=None]')
    synth_options.add_argument('-cr', '--ctx-radius', type=int, default=(3,5,7), nargs='+',
                               help='context radii to use when extracting patches [Default=(3,5,7)]')
    synth_options.add_argument('-th', '--threshold', type=int, default=0,
                               help='threshold for foreground and background (above is foreground) [Default=0]')
    synth_options.add_argument('-pd', '--poly-deg', type=int, default=None,
                               help='degree of polynomial features derived from extracted patches '
                                    '(None means do not use polynomial features) [Default=None]')
    synth_options.add_argument('--mean', action='store_true', default=False,
                               help='learn to take the mean value of input patch to the mean value of output patches')
    synth_options.add_argument('--use-xyz', action='store_true', default=False,
                               help='use the x,y,z coordinates of voxels as features')

    regr_options = parser.add_argument_group('Regressor Options')
    regr_options.add_argument('-n', '--n-jobs', type=int, default=-1,
                              help='number of processors to use (-1 is all processors) [Default=-1]')
    regr_options.add_argument('-msl', '--min-samp-leaf', type=int, default=5,
                              help='minimum number of samples in each leaf in rf (see min_samples_leaf) [Default=5]')
    regr_options.add_argument('-nt', '--n-trees', type=int, default=60,
                              help='number of trees in rf or xg (see n_estimators) [Default=60]')
    regr_options.add_argument('-mf', '--max-features', default=(1.0/3.0),
                              help='proportion of features to use in rf (see max_features) [Default=1/3]')
    regr_options.add_argument('-md', '--max-depth', type=int, default=None,
                              help='maximum tree depth in rf or xg [Default=None (3 for xg)]')
    regr_options.add_argument('-nr', '--num-restarts', type=int, default=8,
                              help='number of restarts for mlr (since finds local optimum) [Default=8]')
    regr_options.add_argument('-mi', '--max-iterations', type=int, default=20,
                              help='maximum number of iterations for mlr and mlp [Default=20]')
    regr_options.add_argument('-hls', '--hidden-layer-sizes', type=int, nargs='+', default=(100,),
                              help='number of neurons in each hidden layer for mlp [Default=(100,)]')
    regr_options.add_argument('-rs', '--random-seed', default=0,
                              help='set random seed for reproducibility [Default=0]')
    return parser


def main(args=None):
    args = arg_parser().parse_args(args)
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
        if args.regr_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            regr = RandomForestRegressor(n_jobs=args.n_jobs, min_samples_leaf=args.min_samp_leaf, n_estimators=args.n_trees,
                                         max_features=args.max_features, max_depth=args.max_depth,
                                         random_state=args.random_seed, verbose=1 if args.verbosity >= 2 else 0)
            flatten = True
        elif args.regr_type == 'xg':
            from xgboost import XGBRegressor
            regr = XGBRegressor(n_jobs=args.n_jobs, n_estimators=args.n_trees, random_state=args.random_seed,
                                max_depth=3 if args.max_depth is None else args.max_depth,
                                silent=False if args.verbosity >=2 else True)
            flatten = True
        elif args.regr_type == 'pr':
            from sklearn.linear_model import LinearRegression
            regr = LinearRegression(n_jobs=args.n_jobs, fit_intercept=True if args.poly_deg is None else False)
            flatten = False
        elif args.regr_type == 'mlr':
            from synthit.synth.mlr import LinearRegressionMixture
            regr = LinearRegressionMixture(3, num_restarts=args.num_restarts, num_workers=args.n_jobs,
                                           max_iterations=args.max_iterations, threshold=args.threshold)
            args.poly_deg = 1 if args.poly_deg is None else args.poly_deg  # hack to get bias term included in features
            flatten = True
        elif args.regr_type == 'mlp':
            from sklearn.neural_network import MLPRegressor
            regr = MLPRegressor(hidden_layer_sizes=args.hidden_layer_sizes, max_iter=args.max_iterations,
                                random_state=args.random_seed, verbose=True if args.verbosity >= 2 else False)
            flatten = True
        else:
            raise SynthError('Invalid regressor type: {}. {{rf, xg, pr, mlr, mlp}} are the only supported options.'.format(args.regr_type))
        logger.debug(regr)
        ps = PatchSynth(regr, args.patch_size, args.n_samples, args.ctx_radius, args.threshold, args.poly_deg,
                        args.mean, args.full_patch, flatten, args.use_xyz)
        source = [ps.image_list(sd) for sd in args.source_dir]
        target = ps.image_list(args.target_dir)
        if any([len(source_) != len(target) for source_ in source]):
            raise SynthError('Number of source and target images must be equal.')
        if args.mask_dir is not None:
            masks = ps.image_list(args.mask_dir)
            if len(masks) != len(target):
                raise SynthError('If masks are provided, the number of masks must be equal to the number of images.')
            source = [[src * mask for (src, mask) in zip(source_, masks)] for source_ in source]
            target = [tgt * mask for (tgt, mask) in zip(target, masks)]
        else:
            masks = [None] * len(target)
        if not args.cross_validate:
            ps.fit(source, target, masks)
            outfile = 'trained_model.pkl' if args.output is None else args.output
            logger.info('Saving trained model: {}'.format(outfile))
            joblib.dump(ps, outfile)
        else:
            for i in range(len(target)):
                src = [[src_ for k, src_ in enumerate(source_) if i != k] for source_ in source]
                tgt = [tgt_ for k, tgt_ in enumerate(target) if i != k]
                msk = [msk_ for k, msk_ in enumerate(masks) if i != k]
                ps.fit(src, tgt, msk)
                if args.output is not None:
                    name, ext = os.path.splitext(args.output)
                    outfile = name + '_{}'.format(i) + ext
                else:
                    outfile = 'trained_model_{}.pkl'.format(i)
                logger.info('Saving trained model: {}'.format(outfile))
                joblib.dump(ps, outfile)
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

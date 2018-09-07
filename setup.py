#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Module installs synthit package
Can be run via command: python setup.py install (or develop)

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Jun 20, 2018
"""

from setuptools import setup, find_packages
from sys import platform


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

if platform == "linux" or platform == "linux32":
    antspy = "https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl"
elif platform == "darwin":
    try:
        import ants
        antspy = ""
    except ImportError:
        raise Exception('On OS X you need to build ANTsPy from source before installing the synthit package. '
                        'See the "install ANTsPy" section of create_env.sh for the necessary commands.')
else:
    raise Exception('synthit package only supports linux and OS X')

args = dict(
    name='synthit',
    version='0.1.0',
    description="Synthesize MR and CT brain images",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://gitlab.com/jcreinhold/synthit',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    entry_points = {
        'console_scripts': ['directory-view=synthit.exec.directory_view:main',
                            'synth-quality=synthit.exec.synth_quality:main',
                            'nn-train=synthit.exec.nn_train:main',
                            'nn-predict=synthit.exec.nn_predict:main',
                            'synth-train=synthit.exec.synth_train:main',
                            'synth-predict=synthit.exec.synth_predict:main',]
    },
    keywords="mr image synthesis",
    dependency_links=[antspy]
)

setup(install_requires=['antspy',
                        'matplotlib',
                        'numpy',
                        'scikit-learn',
                        'scikit-image',
                        'scipy',
                        'torch',
                        'torchvision',
                        'xgboost'], **args)

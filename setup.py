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


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='synthit',
    version='1.0.0',
    description="Synthesize MR and CT brain images",
    long_description=readme,
    author='Jacob Reinhold',
    author_email='jacob.reinhold@jhu.edu',
    url='https://gitlab.com/jcreinhold/synthit',
    license=license,
    packages=find_packages(exclude=('tests', 'tutorials', 'docs')),
    entry_points = {
        'console_scripts': ['synth-train=synthit.exec.synth_train:main',
                            'synth-predict=synthit.exec.synth_predict:main',]
    },
    keywords="mr image synthesis",
)

setup(install_requires=['numpy',
                        'scikit-learn',
                        'scikit-image',
                        'scipy'], **args)

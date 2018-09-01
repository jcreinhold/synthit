synthit
=======

<!---[![Build Status](https://travis-ci.org/jcreinhold/intensity-normalization.svg?branch=master)](https://travis-ci.org/jcreinhold/intensity-normalization)
[![Coverage Status](https://coveralls.io/repos/github/jcreinhold/intensity-normalization/badge.svg?branch=master)](https://coveralls.io/github/jcreinhold/intensity-normalization?branch=master)
[![Documentation Status](https://readthedocs.org/projects/intensity-normalization/badge/?version=latest)](http://intensity-normalization.readthedocs.io/en/latest/?badge=latest)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)-->

This package contains code to synthesize MR neuro images (e.g., learn how to synthesize T2-w images from T1-w images).

[Link to main Gitlab Repository](https://gitlab.com/jcreinhold/synthit)

Requirements
------------

- antspy
- matplotlib
- numpy
- pyro
- scikit-learn
- scikit-image
- pytorch
- xgboost

Basic Usage
-----------

Install from the source directory

    python setup.py install
    
or (if you actively want to make changes to the package)

    python setup.py develop

Test Package
------------

Unit tests can be run from the main directory as follows:

    nosetests -v --with-coverage --cover-tests --cover-package=synthit tests

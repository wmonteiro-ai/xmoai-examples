#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:18:52 2020

@author: wellington
"""


import setuptools
from pymoo import __version__

__name__ = 'xmoai-examples'
__author__ = 'Wellington R Monteiro'
__url__ = 'https://github.com/wmonteiro92/xmoai-examples'

with open("README.md", "r") as fh:
    print(__version__)
    long_description = fh.read()

setuptools.setup(
    name=__name__, # Replace with your own username
    version=__version__,
    author=__author__,
    author_email="NA",
    description="eXplainable Artificial Intelligence using Multiobjective Optimization (examples)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=__url__,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['xmoai>=0.0.dev1',
                      'scikit-learn>=0.23.1',
                      'xgboost>=1.1.0',
                      'lightgbm>=2.3.0',
                      'seaborn>=0.10.0',
                      'tensorflow>=2.2.0']
)
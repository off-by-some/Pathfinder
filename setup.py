#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from imp import load_source
import sys


setup(
    name='vcfx',
    version=load_source('', 'vcfx/__version__.py').__version__,
    description='A Python 3 Vcard parser.',
    author='Cassidy Bridges',
    author_email='cassidybridges@gmail.com',
    url='http://github.com/pholey/vcfx',
    packages=find_packages('.'),
    install_requires=[
        'tornado',
        'numpy',
    ],
    extras_require={
        'test': ['pytest']
    },
)

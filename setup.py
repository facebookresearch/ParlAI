#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='parlai',
    version='0.1.0',
    description='TBD',
    long_description=readme,
    url='https://github.com/facebook/ParlAI',
    license=license,
    packages=find_packages(exclude=('data', 'docs', 'tests'))
)

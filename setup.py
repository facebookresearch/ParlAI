# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='parlai',
    version='0.1.0',
    description='Unified API for accessing dialog datasets.',
    long_description=readme,
    url='https://github.com/facebook/ParlAI',
    license=license,
    packages=find_packages(exclude=(
        'data', 'docs', 'downloads', 'examples', 'tests')),
    install_requires=['pyzmq'],
)

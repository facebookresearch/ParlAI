# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


from setuptools import setup, find_packages
import sys

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for ParlAI.')

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='parlai',
    version='0.1.0',
    description='Unified API for accessing dialog datasets.',
    long_description=readme,
    url='http://parl.ai/',
    license=license,
    packages=find_packages(exclude=(
        'data', 'docs', 'downloads', 'examples', 'logs', 'tests')),
    install_requires=reqs.strip().split('\n'),
)

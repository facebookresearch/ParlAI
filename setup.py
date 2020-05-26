#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import sys

from setuptools import setup, find_packages

BUILD = ''  # test by setting to ".dev0" if multiple in one day, use ".dev1", ...
DATE = datetime.date.today().isoformat().replace('-', '')

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python >=3.6 is required for ParlAI.')

with open('README.md', encoding="utf8") as f:
    # strip the header and badges etc
    readme = f.read().split('--------------------')[-1]

with open('requirements.txt') as f:
    reqs = f.read()


if __name__ == '__main__':
    setup(
        name='parlai',
        version='0.1.{DATE}{BUILD}'.format(DATE=DATE, BUILD=BUILD),
        description='Unified platform for dialogue research.',
        long_description=readme,
        long_description_content_type='text/markdown',
        url='http://parl.ai/',
        python_requires='>=3.6',
        scripts=['bin/parlai'],
        packages=find_packages(
            exclude=('data', 'docs', 'examples', 'tests', 'parlai_internal',)
        ),
        install_requires=reqs.strip().split('\n'),
        include_package_data=True,
        package_data={'': ['*.txt', '*.md']},
        entry_points={"flake8.extension": ["PAI = parlai.utils.flake8:ParlAIChecker"]},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Natural Language :: English",
        ],
    )

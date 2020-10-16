#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys

from setuptools import setup, find_packages

VERSION = '0.9.4'  # if you update, update parlai/__init__.py too!

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python >=3.6 is required for ParlAI.')

with open('README.md', encoding="utf8") as f:
    # strip the header and badges etc
    readme = f.read().split('--------------------')[-1]

with open('requirements.txt') as f:
    reqs = []
    for line in f:
        line = line.strip()
        reqs.append(line.split('==')[0])


if __name__ == '__main__':
    setup(
        name='parlai',
        version=VERSION,
        description='Unified platform for dialogue research.',
        long_description=readme,
        long_description_content_type='text/markdown',
        url='http://parl.ai/',
        python_requires='>=3.6',
        packages=find_packages(
            exclude=('data', 'docs', 'examples', 'tests', 'parlai_internal*')
        ),
        install_requires=reqs,
        include_package_data=True,
        package_data={'': ['*.txt', '*.md']},
        entry_points={
            "flake8.extension": ["PAI = parlai.utils.flake8:ParlAIChecker"],
            "console_scripts": ["parlai=parlai.__main__:main"],
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Natural Language :: English",
        ],
    )

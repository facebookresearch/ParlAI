#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup, find_packages
import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python >=3.6 is required for ParlAI.')

with open('README.md', encoding="utf8") as f:
    readme = f.read()

with open('requirements.txt') as f:
    reqs = f.read()


if __name__ == '__main__':
    setup(
        name='parlai',
        version='0.1.20200408',
        description='Unified platform for dialogue research.',
        long_description=readme,
        long_description_content_type='text/markdown',
        url='http://parl.ai/',
        python_requires='>=3.6',
        packages=find_packages(
            exclude=('data', 'docs', 'examples', 'tests', 'parlai_internal',)
        ),
        install_requires=reqs.strip().split('\n'),
        include_package_data=True,
        entry_points={"flake8.extension": ["PAI = parlai.utils.flake8:ParlAIChecker"]},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Natural Language :: English",
        ],
    )

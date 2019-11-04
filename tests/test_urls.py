#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import unittest
import importlib
import os
import warnings
import sys

TO_SKIP = [
    './parlai/tasks/__pycache__',
    './parlai/tasks/interactive',
    './parlai/tasks/fromfile',
    './parlai/tasks/taskntalk',
    './parlai/tasks/integration_tests',
    './parlai/tasks/dialog_babi_plus',
    './parlai/tasks/decanlp',
]

SPECIFIC_BUILDS = {
    './parlai/tasks/opensubtitles': ['build_2009', 'build_2018'],
    './parlai/tasks/coco_caption': ['build_2014', 'build_2015', 'build_2017'],
}


class TestUtils(unittest.TestCase):
    def test_http_response(self):
        sys.path.insert(0, './lib')
        tasks = [f.path for f in os.scandir('./parlai/tasks') if f.is_dir()]
        for task in tasks:
            if task in TO_SKIP:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                warnings.simplefilter("ignore", DeprecationWarning)
                if task in SPECIFIC_BUILDS:
                    for build in SPECIFIC_BUILDS[task]:
                        mod = importlib.import_module(
                            (task[2:].replace('/', '.') + '.' + build)
                        )
                else:
                    mod = importlib.import_module(
                        (task[2:].replace('/', '.') + '.build')
                    )
                for f in mod.RESOURCES:
                    with self.subTest(f"{task}: {f.url}"):
                        f.check_header()


if __name__ == '__main__':
    unittest.main()

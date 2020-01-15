#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import unittest
import importlib
from parlai.tasks.task_list import task_list

SPECIFIC_BUILDS = {
    'opensubtitles': ['build_2009', 'build_2018'],
    'coco_caption': ['build_2014', 'build_2015', 'build_2017'],
    'dialog_babi_plus': [],
}


class TestUtils(unittest.TestCase):
    def test_http_response(self):
        tasks = set(
            task['task']
            if ':' not in task['task']
            else task['task'][: task['task'].index(':')]
            for task in task_list
        )
        for task in sorted(tasks):
            if task in SPECIFIC_BUILDS:
                for build in SPECIFIC_BUILDS[task]:
                    mod = importlib.import_module(
                        ('parlai.tasks.' + task + '.' + build)
                    )
                    for f in mod.RESOURCES:
                        with self.subTest(f"{task}: {f.url}"):
                            f.check_header()
            else:
                try:
                    mod = importlib.import_module(('parlai.tasks.' + task + '.build'))
                    file_list = mod.RESOURCES
                except (ModuleNotFoundError, AttributeError):
                    continue
                for f in file_list:
                    with self.subTest(f"{task}: {f.url}"):
                        f.check_header()


if __name__ == '__main__':
    unittest.main()

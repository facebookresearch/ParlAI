#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import importlib.util
import unittest

import parlai.core.teachers as teach_module
from parlai.scripts.verify_data import verify, setup_args
from utils import git_changed_files

KEYS = ['missing_text', 'missing_labels', 'empty_label_candidates']


class TestNewTasks(unittest.TestCase):
    """Make sure any changes to tasks pass verify_data test."""

    def test_verify_data(self):
        parser = setup_args()
        opt = parser.parse_args(print_args=False)
        changed_files = git_changed_files()
        changed_task_files = []
        for file in changed_files:
            if 'parlai/tasks' in file and 'README' not in file:
                changed_task_files.append(file)

        if not changed_task_files:
            return

        for file in changed_task_files:
            task = file.split('/')[-2]
            module_name = "%s.tasks.%s.agents" % ('parlai', task)
            task_module = importlib.import_module(module_name)
            base_teachers = dir(teach_module) + ['PytorchDataTeacher']
            subtasks = [':'.join([task, x]) for x in dir(task_module) if
                        ('teacher' in x.lower() and x not in base_teachers)]

            for subt in subtasks:
                opt['task'] = subt
                text, log = verify(opt, print_parser=False)
                for key in KEYS:
                    self.assertEqual(
                        log[key],
                        0,
                        'There are {} {} in this task.'.format(
                            log[key],
                            log
                        ))


if __name__ == '__main__':
    unittest.main()

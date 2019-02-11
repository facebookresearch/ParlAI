#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest

import parlai.core.teachers as teach_module
from parlai.scripts.verify_data import verify, setup_args
import parlai.core.testing_utils as testing_utils

KEYS = ['missing_text', 'missing_labels', 'empty_label_candidates']
BASE_TEACHERS = dir(teach_module) + ['PytorchDataTeacher']


class TestNewTasks(unittest.TestCase):
    """Make sure any changes to tasks pass verify_data test."""

    def test_verify_data(self):
        parser = setup_args()
        opt = parser.parse_args(print_args=False)
        changed_files = testing_utils.git_changed_files()
        changed_task_files = []
        for file in changed_files:
            if (
                'parlai/tasks' in file and
                'README' not in file and
                'task_list.py' not in file
            ):
                changed_task_files.append(file)

        if not changed_task_files:
            return

        for file in changed_task_files:
            task = file.split('/')[-2]
            module_name = "%s.tasks.%s.agents" % ('parlai', task)
            task_module = importlib.import_module(module_name)
            subtasks = [
                ':'.join([task, x])
                for x in dir(task_module)
                if ('teacher' in x.lower() and x not in BASE_TEACHERS)
            ]

            for subt in subtasks:
                opt['task'] = subt
                with testing_utils.capture_output() as _:
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

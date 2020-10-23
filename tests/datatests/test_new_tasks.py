#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import traceback
import unittest

import parlai.core.teachers as teach_module
from parlai.scripts.verify_data import verify, setup_args
import parlai.utils.testing as testing_utils

KEYS = ['missing_text', 'missing_labels', 'empty_string_label_candidates']
BASE_TEACHERS = dir(teach_module) + ['CandidateBaseTeacher']


class TestNewTasks(unittest.TestCase):
    """
    Make sure any changes to tasks pass verify_data test.
    """

    def test_verify_data(self):
        parser = setup_args()
        opt = parser.parse_args([])
        changed_task_files = [
            fn
            for fn in testing_utils.git_changed_files()
            if testing_utils.is_new_task_filename(fn)
        ]
        if not changed_task_files:
            return

        found_errors = False
        for file in changed_task_files:
            task = file.split('/')[-2]
            module_name = "%s.tasks.%s.agents" % ('parlai', task)
            task_module = importlib.import_module(module_name)
            subtasks = [
                ':'.join([task, x])
                for x in dir(task_module)
                if x.endswith('Teacher') and x not in BASE_TEACHERS
            ]

            if testing_utils.is_this_circleci():
                if len(subtasks) == 0:
                    continue

                self.fail(
                    'test_verify_data plays poorly with CircleCI. Please run '
                    '`python tests/datatests/test_new_tasks.py` locally and '
                    'paste the output in your pull request.'
                )

            for subt in subtasks:
                parser = setup_args()
                opt = parser.parse_args(args=['--task', subt])
                opt['task'] = subt
                try:
                    with testing_utils.capture_output():
                        verify_results = verify(opt)
                except Exception:
                    found_errors = True
                    traceback.print_exc()
                    print("Got above exception in {}".format(subt))
                for key in KEYS:
                    if verify_results[key] != 0:
                        print(
                            'There are {} {} in {}.'.format(
                                verify_results[key], key, subt
                            )
                        )
                        found_errors = True

        if found_errors:
            self.fail(
                "Please fix the above listed errors, or describe in the PR why "
                "you do not expect them to pass."
            )


if __name__ == '__main__':
    unittest.main()

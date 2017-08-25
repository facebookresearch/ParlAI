# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import os
import unittest


class TestInit(unittest.TestCase):
    """Make sure the package is alive."""

    def test_tasklist(self):
        from parlai.tasks.task_list import task_list
        from parlai.core.params import ParlaiParser
        opt = ParlaiParser().parse_args()

        a = set((t['task'].split(':')[0] for t in task_list))

        task_dir = os.path.join(opt['parlai_home'], 'parlai', 'tasks')
        b = set((t for t in os.listdir(task_dir) if '.' not in t and t != '__pycache__' and t != 'fromfile'))
        if a != b:
            not_in_b = a - b
            not_in_a = b - a
            error_msg = ''
            if len(not_in_b) > 0:
                error_msg += '\nThe following tasks are in the task list but do not have directories in parlai/tasks/: ' + str(not_in_b)
            if len(not_in_a) > 0:
                error_msg += '\nThe following tasks are in parlai/tasks/ but do not match anything in parlai/tasks/task_list.py: ' + str(not_in_a)
            raise RuntimeError(error_msg)


if __name__ == '__main__':
    unittest.main()

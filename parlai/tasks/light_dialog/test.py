#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.scripts.display_data import setup_args

from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest


class TestDefaultTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'light_dialog'


NUM_EPS = 17075


class TestLightTeacher(unittest.TestCase):
    """
    Tests for the LIGHT Dialogue Teacher.
    """

    def test_counts(self):
        proportions = [0.1, 0.5, 1.0]
        for proportion in proportions:
            all_kwargs = {
                'datatype': 'train',
                'task': 'light_dialog',
                'light_percent_train_exs': proportion,
            }
            parser = setup_args()
            parser.set_defaults(**all_kwargs)
            opt = parser.parse_args([])
            agent = RepeatLabelAgent(opt)
            teacher = create_task(opt, agent).get_task_agent()
            self.assertEqual(teacher.num_episodes(), int(NUM_EPS * proportion))


if __name__ == '__main__':
    unittest.main()

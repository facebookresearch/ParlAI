#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.scripts.display_data import setup_args


class TestWrapper(unittest.TestCase):
    def test_label_to_text_teacher(self):

        # Set up regular teacher
        kwargs = {'task': 'integration_tests:multiturn'}
        parser = setup_args()
        parser.set_defaults(**kwargs)
        opt = parser.parse_args([])
        agent = RepeatLabelAgent(opt)
        regular_world = create_task(opt, agent)

        # Set up label-to-text teacher
        kwargs = {
            'task': 'wrapper:labelToTextTeacher',
            'wrapper_task': 'integration_tests:multiturn',
        }
        parser = setup_args()
        parser.set_defaults(**kwargs)
        opt = parser.parse_args([])
        agent = RepeatLabelAgent(opt)
        label_to_text_world = create_task(opt, agent)

        num_examples = 0
        while num_examples < 5:
            regular_world.parley()
            regular_example = regular_world.get_acts()[0]
            label_to_text_world.parley()
            label_to_text_example = label_to_text_world.get_acts()[0]
            self.assertEqual(
                label_to_text_example['text'], regular_example['labels'][0]
            )
            self.assertEqual(label_to_text_example['labels'], [''])
            num_examples += 1


if __name__ == '__main__':
    unittest.main()

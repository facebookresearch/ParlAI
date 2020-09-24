#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import unittest
from parlai.tasks.convai_chitchat.agents import DefaultTeacher

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


class TeacherTest(unittest.TestCase):
    def test_data_generator(self):
        """
        Checks:
        - data_generator should create two episodes from the one dialogue entry
        - data_generator should put the first user on one side of the
          conversation and the second user on the other side.
        - data_generator should correct parse dialog from dict
        - data_generator should works with context
        """
        with open(os.path.join(__location__, 'dialogs.fixture.json')) as data_file:
            dialogs = json.load(data_file)

        correct_items = [
            ((dialogs[0]['context'], ['']), True),
            (('Hello there!', ["Hey, what's up?"]), False),
            (('Nothing much. You?', ['Same.']), False),
            (('Some context', ['Hello there!']), True),
            (("Hey, what's up?", ['Nothing much. You?']), False),
        ]
        self.assertListEqual(
            [i for i in DefaultTeacher._data_generator(dialogs)], correct_items
        )


if __name__ == '__main__':
    unittest.main()

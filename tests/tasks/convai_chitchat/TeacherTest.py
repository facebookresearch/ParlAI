# Copyright (c) 2017-present, Moscow Institute of Physics and Technology.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

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
             - data_generator should put the first user on one side of the conversation and the second user
             on the other side.
             - data_generator should correct parse dialog from dict
             - data_generator should works with context
        """
        with open(os.path.join(__location__, 'dialogs.fixture.json')) as data_file:
            dialogs = json.load(data_file)

        correct_items = [
            ((dialogs[0]['context'], ['']), False),
            (('Hello there!', ["Hey, what's up?"]), False),
            (('Nothing much. You?', ['Same.']), True),
            (('Some context', ['Hello there!']), False),
            (("Hey, what's up?", ['Nothing much. You?']), True)
        ]
        self.assertListEqual([i for i in DefaultTeacher._data_generator(dialogs)], correct_items)


if __name__ == '__main__':
    unittest.main()

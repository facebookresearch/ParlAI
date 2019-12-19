#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test Teachers that are not in core/teachers.py but are in task specific folders.
test_teachers.py tests teachers that are in core/teachers.py

"""

import os
import numpy as np
import json
import unittest
from parlai.utils import testing as testing_utils
import regex as re


class BaseTeacherTest(unittest.TestCase):
    """
    Tests the number of examples, number of episodes, and that the first N
    examples matches that in a file in test_data called
    first_n_{teacher_name}_{count}.json
    
    Use the utility function write_first_n_examples_to_file() when first generating the data so that it will use a numpy seed
    TODO: how to test epoch_done(?)
    """

    def __init__(
        self,
        teacher_name,
        teacher_opt=None,
        num_examples=-1,
        num_episodes=-1,
        first_n_count=10,
    ):
        # In case the data is shuffled
        NUMPY_SEED = 42
        np.random.seed(NUMPY_SEED)

        self.__teacher_name = teacher_name
        self.__teacher_opt = teacher_opt
        self.teacher = BaseTeacherTest.get_teacher_from_name(
            self.__teacher_name, self.__teacher_opt
        )
        self.expected_num_examples = num_examples
        self.expected_num_episodes = num_episodes
        self.expected_first_n_count = first_n_count

    @staticmethod
    def get_teacher_from_name(teacher_name, teacher_opt):
        teacher_class = None  # TODO
        return teacher_class(teacher_opt)

    def test_num_examples(self):
        self.assertEqual(self.expected_num_examples, self.teacher.num_examples())

    def test_num_episodes(self):
        self.assertEqual(self.expected_num_episodes, self.teacher.num_episodes())

    def test_first_n(self):
        cwd = os.getcwd()
        test_data_for_teacher = os.path.join(
            cwd, 'test_data', f'first_n_{self.teacher_name}_{self.expected_first_n_count}.json'
        )
        self.assertTrue(
            os.path.isfile(test_data_for_teacher),
            msg=f'Test data file for teacher {self.teacher_name} ({test_data_for_teacher}) did not exist.',
        )
        first_n_examples = []
        with open(test_data_for_teacher, 'r') as data_file:
            expected_first_n_examples = json.loads(data_file.read())
        self.assertEqual(self.expected_first_n_count, len(first_n_examples))
        actual_n_examples = BaseTeacherTest.get_first_n_examples(
            self.teacher, self.expected_first_n_count
        )
        self.assertEqual(actual_n_examples, expected_first_n_examples)

    @staticmethod
    def get_first_n_examples(teacher, n):
        """ Returns array of length <n> tuples of [(text, first label), ...]"""
        examples_count = 0
        examples = []
        for episode_idx in range(teacher.num_episodes()):
            entry_idx = 0
            while True:
                act = teacher.get(episode_idx, entry_idx=entry_idx)
                examples.append((act['text'], act['labels'][0]))
                examples_count += 1
                if act['episode_done']:
                    break
                else:
                    entry_idx += 1
        return examples


def write_first_n_examples_to_file(self, teacher_name, teacher_opt=None, n=10):
    """
    Utility function to be used if you are first generating the data for the test. Will write
    it to test_data/
    """

    # In case the data is shuffled
    NUMPY_SEED = 42
    np.random.seed(NUMPY_SEED)

    cwd = os.getcwd()
    test_data_file_for_teacher = os.path.join(
        cwd, 'test_data', f'first_n_{teacher_name}_{n}.json'
    )
    teacher = BaseTeacherTest.get_teacher_from_name(teacher_name, teacher_opt)
    first_n_examples = BaseTeacherTest.get_first_n_examples(teacher, n)
    with open(test_data_file_for_teacher, 'r') as f:
        f.write(json.dumps(first_n_examples))


if __name__ == '__main__':
    unittest.main()

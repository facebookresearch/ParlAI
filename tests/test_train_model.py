#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic tests that ensure train_model.py behaves in predictable ways.
"""

import unittest
import parlai.core.testing_utils as testing_utils


class TestTrainModel(unittest.TestCase):
    def test_fast_final_eval(self):
        stdout, valid, test = testing_utils.train_model(
            {
                'task': 'integration_tests',
                'validation_max_exs': 10,
                'model': 'repeat_label',
                'short_final_eval': True,
                'num_epochs': 1.0,
            }
        )
        self.assertEqual(valid['exs'], 10, 'Validation exs is wrong')
        self.assertEqual(test['exs'], 10, 'Test exs is wrong')

    def test_multitasking_metrics(self):
        stdout, valid, test = testing_utils.train_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': True,
            }
        )

        task1_acc = valid['tasks']['integration_tests:candidate']['accuracy']
        task2_acc = valid['tasks']['integration_tests:multiturnCandidate']['accuracy']
        total_acc = valid['accuracy']
        # task 2 is 4 times the size of task 1
        self.assertAlmostEqual(
            total_acc,
            (task1_acc + 4 * task2_acc) / 5,
            4,
            'Task accuracy is averaged incorrectly',
        )

        stdout, valid, test = testing_utils.train_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': False,
            }
        )
        task1_acc = valid['tasks']['integration_tests:candidate']['accuracy']
        task2_acc = valid['tasks']['integration_tests:multiturnCandidate']['accuracy']
        total_acc = valid['accuracy']
        # metrics should be averaged equally across tasks
        self.assertAlmostEqual(
            total_acc,
            (task1_acc + task2_acc) / 2,
            4,
            'Task accuracy is averaged incorrectly',
        )


if __name__ == '__main__':
    unittest.main()

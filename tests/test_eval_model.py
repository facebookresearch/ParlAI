#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from examples.eval_model import setup_args

import ast
import unittest
import parlai.core.testing_utils as testing_utils


class TestEvalModel(unittest.TestCase):
    """Basic tests on the eval_model.py example."""

    def test_output(self):
        """Test output of running eval_model"""
        parser = setup_args()
        parser.set_defaults(
            task='integration_tests',
            model='repeat_label',
            datatype='valid',
            num_examples=5,
            display_examples=False,
        )

        opt = parser.parse_args(print_args=False)
        str_output, valid, test = testing_utils.eval_model(opt)
        self.assertGreater(len(str_output), 0, "Output is empty")

        # decode the output
        scores = str_output.split("\n---\n")

        for i in range(1, len(scores)):
            score = ast.literal_eval(scores[i])
            # check totals
            self.assertEqual(score['exs'], i, "Total is incorrect")
            # accuracy should be one
            self.assertEqual(
                'accuracy' in score, True, "Accuracy is missing from default"
            )
            self.assertEqual(score['accuracy'], 1, "Accuracy != 1")
            self.assertEqual(
                'rouge-1' in score, False, "Rouge is in the default metrics"
            )

    def test_metrics_all(self):
        """Test output of running eval_model"""
        parser = setup_args()
        parser.set_defaults(
            task='integration_tests',
            model='repeat_label',
            datatype='valid',
            num_examples=5,
            display_examples=False,
            metrics='all',
        )

        opt = parser.parse_args(print_args=False)
        str_output, valid, test = testing_utils.eval_model(opt)
        self.assertGreater(len(str_output), 0, "Output is empty")

        # decode the output
        scores = str_output.split("\n---\n")

        for i in range(1, len(scores)):
            score = ast.literal_eval(scores[i])
            # check totals
            self.assertEqual(score['exs'], i, "Total is incorrect")
            # accuracy should be one
            self.assertEqual('accuracy' in score, True, "Accuracy is missing from all")
            self.assertEqual(score['accuracy'], 1, "Accuracy != 1")
            self.assertEqual('rouge-1' in score, True, "Rouge is missing from all")
            self.assertEqual(score['rouge-1'], 1, 'rouge1 != 1')
            self.assertEqual(score['rouge-2'], 1, 'rouge-2 != 1')
            self.assertEqual(score['rouge-L'], 1, 'rouge-L != 1')

    def test_metrics_select(self):
        """Test output of running eval_model"""
        parser = setup_args()
        parser.set_defaults(
            task='integration_tests',
            model='repeat_label',
            datatype='valid',
            num_examples=5,
            display_examples=False,
            metrics='accuracy,rouge',
        )

        opt = parser.parse_args(print_args=False)
        str_output, valid, test = testing_utils.eval_model(opt)
        self.assertGreater(len(str_output), 0, "Output is empty")

        # decode the output
        scores = str_output.split("\n---\n")

        for i in range(1, len(scores)):
            score = ast.literal_eval(scores[i])
            # check totals
            self.assertEqual(score['exs'], i, "Total is incorrect")
            # accuracy should be one
            self.assertEqual(
                'accuracy' in score, True, "Accuracy is missing from selection"
            )
            self.assertEqual(score['accuracy'], 1, "Accuracy != 1")
            self.assertEqual(
                'rouge-1' in score, True, "Rouge is missing from selection"
            )
            self.assertEqual(score['rouge-1'], 1, 'rouge1 != 1')
            self.assertEqual(score['rouge-2'], 1, 'rouge-2 != 1')
            self.assertEqual(score['rouge-L'], 1, 'rouge-L != 1')

    def test_multitasking_metrics(self):
        stdout, valid, test = testing_utils.eval_model(
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

        stdout, valid, test = testing_utils.eval_model(
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

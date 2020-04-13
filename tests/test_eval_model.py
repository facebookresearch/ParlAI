#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.scripts.eval_model import setup_args

import os
import unittest
import parlai.utils.testing as testing_utils


class TestEvalModel(unittest.TestCase):
    """
    Basic tests on the eval_model.py example.
    """

    def test_noevalmode(self):
        """
        Ensure you get an error trying to use eval_model with -dt train.
        """
        with self.assertRaises(ValueError):
            testing_utils.eval_model(
                {'task': 'integration_tests', 'model': 'repeat_label'},
                valid_datatype='train',
            )

    def test_evalmode(self):
        """
        Eval_model with -dt train:evalmode should be okay.
        """
        testing_utils.eval_model(
            {'task': 'integration_tests', 'model': 'repeat_label'},
            valid_datatype='train:evalmode',
        )

    def test_output(self):
        """
        Test output of running eval_model.
        """
        parser = setup_args()
        parser.set_defaults(
            task='integration_tests',
            model='repeat_label',
            datatype='valid',
            num_examples=5,
            display_examples=False,
        )

        opt = parser.parse_args([], print_args=False)
        valid, test = testing_utils.eval_model(opt)

        self.assertEqual(valid['accuracy'], 1)
        self.assertEqual(test['accuracy'], 1)
        self.assertNotIn('rouge-L', valid)
        self.assertNotIn('rouge-L', test)

    def test_metrics_all(self):
        """
        Test output of running eval_model.
        """
        parser = setup_args()
        parser.set_defaults(
            task='integration_tests',
            model='repeat_label',
            datatype='valid',
            num_examples=5,
            display_examples=False,
            metrics='all',
        )

        opt = parser.parse_args([], print_args=False)
        valid, test = testing_utils.eval_model(opt)

        self.assertEqual(valid['accuracy'], 1)
        self.assertEqual(valid['rouge-L'], 1)
        self.assertEqual(valid['rouge-1'], 1)
        self.assertEqual(valid['rouge-2'], 1)
        self.assertEqual(test['accuracy'], 1)
        self.assertEqual(test['rouge-L'], 1)
        self.assertEqual(test['rouge-1'], 1)
        self.assertEqual(test['rouge-2'], 1)

    def test_metrics_select(self):
        """
        Test output of running eval_model.
        """
        parser = setup_args()
        parser.set_defaults(
            task='integration_tests',
            model='repeat_label',
            datatype='valid',
            num_examples=5,
            display_examples=False,
            metrics='accuracy,rouge',
        )

        opt = parser.parse_args([], print_args=False)
        valid, test = testing_utils.eval_model(opt)

        self.assertEqual(valid['accuracy'], 1)
        self.assertEqual(valid['rouge-L'], 1)
        self.assertEqual(valid['rouge-1'], 1)
        self.assertEqual(valid['rouge-2'], 1)
        self.assertEqual(test['accuracy'], 1)
        self.assertEqual(test['rouge-L'], 1)
        self.assertEqual(test['rouge-1'], 1)
        self.assertEqual(test['rouge-2'], 1)

        self.assertNotIn('bleu-4', valid)
        self.assertNotIn('bleu-4', test)

    def test_multitasking_metrics_macro(self):
        valid, test = testing_utils.eval_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': False,
            }
        )

        task1_acc = valid['integration_tests:candidate/accuracy']
        task2_acc = valid['integration_tests:multiturnCandidate/accuracy']
        total_acc = valid['accuracy']
        # task 2 is 4 times the size of task 1
        self.assertEqual(
            total_acc,
            (task1_acc.value() + task2_acc.value()) * 0.5,
            'Task accuracy is averaged incorrectly',
        )

        valid, test = testing_utils.eval_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': False,
            }
        )
        task1_acc = valid['integration_tests:candidate/accuracy']
        task2_acc = valid['integration_tests:multiturnCandidate/accuracy']
        total_acc = valid['accuracy']

        # metrics are combined correctly
        self.assertEqual(
            total_acc,
            (task1_acc.value() + task2_acc.value()) * 0.5,
            'Task accuracy is averaged incorrectly',
        )

    def test_multitasking_metrics_micro(self):
        valid, test = testing_utils.eval_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': True,
            }
        )

        task1_acc = valid['integration_tests:candidate/accuracy']
        task2_acc = valid['integration_tests:multiturnCandidate/accuracy']
        total_acc = valid['accuracy']
        # task 2 is 4 times the size of task 1
        self.assertEqual(
            total_acc, task1_acc + task2_acc, 'Task accuracy is averaged incorrectly',
        )

        valid, test = testing_utils.eval_model(
            {
                'task': 'integration_tests:candidate,'
                'integration_tests:multiturnCandidate',
                'model': 'random_candidate',
                'num_epochs': 0.5,
                'aggregate_micro': True,
            }
        )
        task1_acc = valid['integration_tests:candidate/accuracy']
        task2_acc = valid['integration_tests:multiturnCandidate/accuracy']
        total_acc = valid['accuracy']

        # metrics are combined correctly
        self.assertEqual(
            total_acc, (task1_acc + task2_acc), 'Task accuracy is averaged incorrectly',
        )

    def test_train_evalmode(self):
        """
        Test that evaluating a model with train:evalmode completes an epoch.
        """
        base_dict = {'model': 'repeat_label', 'datatype': 'train:evalmode'}

        teachers = ['integration_tests:fixed_dialog_candidate', 'integration_tests']
        batchsize = [1, 64]
        for bs in batchsize:
            for teacher in teachers:
                d = base_dict.copy()
                d['task'] = teacher
                d['batchsize'] = bs
                with testing_utils.timeout(time=20):
                    valid, test = testing_utils.eval_model(
                        d, valid_datatype=d['datatype']
                    )
                self.assertEqual(
                    int(valid['exs']),
                    500,
                    f'train:evalmode failed with bs {bs} and teacher {teacher}',
                )

    def test_save_report(self):
        """
        Test that we can save report from eval model.
        """
        with testing_utils.tempdir() as tmpdir:
            save_report = os.path.join(tmpdir, 'report')
            parser = setup_args()
            parser.set_defaults(
                task='integration_tests',
                model='repeat_label',
                datatype='valid',
                num_examples=5,
                display_examples=False,
                save_world_logs=True,
                report_filename=save_report,
            )

            opt = parser.parse_args([], print_args=False)
            valid, test = testing_utils.eval_model(opt)


if __name__ == '__main__':
    unittest.main()

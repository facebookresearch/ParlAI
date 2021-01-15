#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic tests that ensure train_model.py behaves in predictable ways.
"""
import os
import unittest
import parlai.utils.testing as testing_utils
from parlai.core.worlds import create_task
from parlai.core.params import ParlaiParser


class TestTrainModel(unittest.TestCase):
    def test_fast_final_eval(self):
        valid, test = testing_utils.train_model(
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

    def test_multitasking_metrics_micro(self):
        valid, test = testing_utils.train_model(
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
        self.assertEqual(
            total_acc, task1_acc + task2_acc, 'Task accuracy is averaged incorrectly'
        )

        valid, test = testing_utils.train_model(
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
        # metrics should be averaged equally across tasks
        self.assertEqual(
            total_acc, task1_acc + task2_acc, 'Task accuracy is averaged incorrectly'
        )

    def test_multitasking_metrics_macro(self):
        valid, test = testing_utils.train_model(
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
        self.assertEqual(
            total_acc,
            0.5 * (task1_acc.value() + task2_acc.value()),
            'Task accuracy is averaged incorrectly',
        )

        valid, test = testing_utils.train_model(
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
        # metrics should be averaged equally across tasks
        self.assertEqual(
            total_acc,
            0.5 * (task1_acc.value() + task2_acc.value()),
            'Task accuracy is averaged incorrectly',
        )

    def test_multitasking_id_overlap(self):
        with self.assertRaises(AssertionError) as context:
            pp = ParlaiParser()
            opt = pp.parse_args(['--task', 'integration_tests,integration_tests'])
            self.world = create_task(opt, None)
            self.assertTrue(
                'teachers have overlap in id integration_tests.'
                in str(context.exception)
            )

    def _test_opt_step_opts(self, update_freq: int):
        """
        Test -tstep, -vstep, -lstep.

        :param update_freq:
            update frequency

        We copy train_model from testing_utils to directly access train loop.
        """
        import parlai.scripts.train_model as tms

        num_train_steps = 1001
        num_validations = 10
        num_logs = 100

        def get_tl(tmpdir):
            opt = {
                'task': 'integration_tests',
                'model': 'parlai.agents.test_agents.test_agents:MockTrainUpdatesAgent',
                'model_file': os.path.join(tmpdir, 'model'),
                'dict_file': os.path.join(tmpdir, 'model.dict'),
                # step opts
                'max_train_steps': num_train_steps,
                'validation_every_n_steps': int(num_train_steps / num_validations),
                'log_every_n_steps': int(num_train_steps / num_logs),
                'update_freq': update_freq,
            }
            parser = tms.setup_args()
            parser.set_params(**opt)
            popt = parser.parse_args([])
            for k, v in opt.items():
                popt[k] = v
            return tms.TrainLoop(popt)

        with testing_utils.capture_output(), testing_utils.tempdir() as tmpdir:
            tl = get_tl(tmpdir)
            valid, _ = tl.train()

        self.assertEqual(
            tl.valid_reports[-1]['total_train_updates'], num_train_steps - 1
        )
        self.assertEqual(len(tl.valid_reports), num_validations)
        self.assertEqual(
            len(tl.train_reports), num_logs + num_validations
        )  # log every valid as well

    def test_opt_step(self):
        self._test_opt_step_opts(1)

    def test_opt_step_update_freq_2(self):
        self._test_opt_step_opts(2)


if __name__ == '__main__':
    unittest.main()

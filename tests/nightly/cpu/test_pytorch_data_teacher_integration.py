#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.scripts.display_data import setup_args as display_setup_args, display_data
from parlai.core.agents import create_task_agent_from_taskname

import unittest
import parlai.core.testing_utils as testing_utils

parser_defaults = {
    'model': 'seq2seq',
    'pytorch_teacher_task': 'integration_tests:nocandidate',
    'batchsize': 16,
    'hiddensize': 16,
    'attention': 'general',
    'rnn_class': 'gru',
    'no_cuda': True,
    'learningrate': 1,
    'embeddingsize': 16,
    'dropout': 0.0,
    'gradient_clip': 1.0,
    'lookuptable': 'all',
    'num_epochs': 50,
    'validation_every_n_epochs': 5,
    'log_every_n_secs': 1,
    'batch_length_range': 5,
}


def solved_task(str_output, valid, test):
    return (
        valid['ppl'] < 1.3 and
        test['ppl'] < 1.3 and
        valid['accuracy'] > 0.95 and
        test['accuracy'] > 0.95
    )


class TestPytorchDataTeacher(unittest.TestCase):
    """Various Integration tests for PytorchDataTeacher"""

    def _pyt_train(self, datatype):
        """
        Integration test: ensure that pytorch data teacher can successfully
        teach Seq2Seq model to fully solve the babi:task10k:1 task.

        The Seq2Seq model can solve the babi:task10k:1 task with the normal
        ParlAI setup, and thus should be able to with a PytorchDataTeacher
        """
        defaults = parser_defaults.copy()
        defaults['datatype'] = datatype
        defaults['shuffle'] = True  # for train:stream
        str_output, valid, test = testing_utils.train_model(defaults)
        self.assertTrue(
            solved_task(str_output, valid, test),
            'Teacher could not teach seq2seq with args: {}; here is str_output: {}'
            .format(defaults, str_output)
        )

    def test_pyt_train(self):
        self._pyt_train('train')

    def test_pyt_train_stream(self):
        self._pyt_train('train:stream')

    def test_pyt_train_stream_ordered(self):
        self._pyt_train('train:stream:ordered')

    def test_pyt_preprocess_train(self):
        """
        Test that the preprocess functionality works with the PytorchDataTeacher
        with a sample TorchAgent (here, the Seq2seq model).

        This tests whether an agent can train to completion with
        these preprocessed examples
        """
        defaults = parser_defaults.copy()
        defaults['datatype'] = 'train'
        defaults['pytorch_preprocess'] = True
        str_output, valid, test = testing_utils.train_model(defaults)
        self.assertTrue(
            solved_task(str_output, valid, test),
            'Teacher could not teach seq2seq with preprocessed obs, output: {}'
            .format(str_output)
        )

    def _pyt_batchsort_train(self, datatype, preprocess):
        """
        Tests the functionality of training with batchsort

        :param datatype:
            (string) datatype to train with
        :param preprocess:
            (bool) whether to preprocess the data
        """
        defaults = parser_defaults.copy()
        defaults['datatype'] = datatype
        defaults['pytorch_preprocess'] = preprocess
        defaults['pytorch_teacher_batch_sort'] = True
        defaults['batchsize'] = 32
        if preprocess:
            defaults['batch_sort_field'] = 'text_vec'
        str_output, valid, test = testing_utils.train_model(defaults)
        self.assertTrue(
            solved_task(str_output, valid, test),
            'Teacher could not teach seq2seq with batch sort '
            'and args {} and output {}'
            .format((datatype, preprocess), str_output)
        )

    def test_pyt_batchsort_train(self):
        self._pyt_batchsort_train('train', False)

    def test_pyt_batchsort_train_stream(self):
        self._pyt_batchsort_train('train:stream', False)

    def test_pyt_batchsort_train_preprocess(self):
        self._pyt_batchsort_train('train', True)

    def test_pytd_teacher(self):
        """
        Test that the pytorch teacher works with given Pytorch Datasets
        as well
        """
        defaults = parser_defaults.copy()
        defaults['datatype'] = 'train:stream'
        defaults['image_mode'] = 'ascii'

        with testing_utils.capture_output():
            # Get processed act from agent
            parser = display_setup_args()
            defaults['pytorch_teacher_dataset'] = 'integration_tests'
            del defaults['pytorch_teacher_task']
            parser.set_defaults(**defaults)
            opt = parser.parse_args()
            teacher = create_task_agent_from_taskname(opt)[0]
            pytorch_teacher_act = teacher.act()

            parser = display_setup_args()
            defaults['task'] = 'integration_tests'
            del defaults['pytorch_teacher_dataset']
            parser.set_defaults(**defaults)
            opt = parser.parse_args()
            teacher = create_task_agent_from_taskname(opt)[0]
            regular_teacher_act = teacher.act()

        keys = set(pytorch_teacher_act.keys()).intersection(
            set(regular_teacher_act.keys()))
        self.assertTrue(len(keys) != 0)
        for key in keys:
            self.assertTrue(pytorch_teacher_act[key] == regular_teacher_act[key],
                            'PytorchDataTeacher does not have the same value '
                            'as regular teacher for act key: {}. '
                            'Values: {}; {}'.format(
                                key, pytorch_teacher_act[key], regular_teacher_act[key])
                            )

    def test_pyt_multitask(self):
        """
            Unit test for ensuring that PytorchDataTeacher correctly handles
            multitasking.

            This test will iterate through the following scenarios:
                1. 2 `pytorch_teacher_task`s
                2. 1 `pytorch_teacher_task`, 1 regular ParlAI task
                3. 1 `pytorch_teacher_task`, 1 `pytorch_teacher_dataset`
                4. 1 `pytorch_teacher_dataset`, 1 regular ParlAI task
                5. 2 `pytorch_teacher_dataset`s

        """

        def run_display_test(defaults, ep_and_ex_counts):
            with testing_utils.capture_output() as f:
                parser = display_setup_args()
                parser.set_defaults(**defaults)
                opt = parser.parse_args()
                display_data(opt)
            str_output = f.getvalue()
            self.assertTrue(
                '[ loaded {} episodes with a total of {} examples ]'.format(
                    ep_and_ex_counts[0], ep_and_ex_counts[1]
                ) in str_output,
                'PytorchDataTeacher multitasking failed with '
                'following args: {}'.format(opt)
            )

        # task; num eps; num exs
        task1 = 'babi:task1k:1'  # 180, 900
        task2 = 'babi:task1k:2'  # 180, 900
        dataset1 = 'integration_tests'  # 500, 500
        dataset2 = 'integration_tests:NoCandidateTeacherDataset'  # 500, 500

        # Expected example and episode counts
        eps_and_exs_counts = [
            (1800, 1800),
            (1080, 1800),
            (1400, 1400),
            (680, 1400),
            (1000, 1000)
        ]
        defaults = parser_defaults.copy()

        # 1.
        defaults['pytorch_teacher_task'] = '{},{}'.format(task1, task2)
        run_display_test(defaults, eps_and_exs_counts[0])

        # 2.
        defaults['pytorch_teacher_task'] = task1
        defaults['task'] = task2
        run_display_test(defaults, eps_and_exs_counts[1])

        # 3.
        del defaults['task']
        defaults['pytorch_teacher_dataset'] = dataset1
        run_display_test(defaults, eps_and_exs_counts[2])

        # 4.
        del defaults['pytorch_teacher_task']
        defaults['task'] = task1
        run_display_test(defaults, eps_and_exs_counts[3])

        # 5.
        del defaults['task']
        defaults['pytorch_teacher_dataset'] = '{},{}'.format(dataset1, dataset2)
        run_display_test(defaults, eps_and_exs_counts[4])


if __name__ == '__main__':
    unittest.main()

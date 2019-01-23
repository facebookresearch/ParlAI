#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.scripts.display_data import setup_args as display_setup_args, \
    display_data
from parlai.scripts.train_model import TrainLoop, setup_args as train_setup_args
from parlai.core.agents import create_task_agent_from_taskname

import unittest
import tempfile
import io
import os
from contextlib import redirect_stdout

parser_defaults = {
    'model': 'seq2seq',
    'pytorch_teacher_task': 'babi:task10k:1',
    'model_file': '/tmp/tmp_model',
    'batchsize': 32,
    'momentum':  0.9,
    'validation_every_n_secs': 30,
    'batch_length_range': 5,
}


def set_model_file(defaults):
    defaults['model_file'] = os.path.join(tempfile.mkdtemp(), 'model')
    defaults['dict_file'] = defaults['model_file'] + '.dict'


def solved_task(str_output):
    if '[ task solved! stopping. ]' in str_output:
        return True

    acc_string = '\'accuracy\': '
    idx = str_output.rfind(acc_string)
    score = float(str_output[idx+len(acc_string):str_output.index(',', idx)])
    return score >= 0.90


class TestPytorchDataTeacher(unittest.TestCase):
    """Various Integration tests for PytorchDataTeacher"""

    def test_pyt_train(self):
        """
            Integration test: ensure that pytorch data teacher can successfully
            teach Seq2Seq model to fully solve the babi:task10k:1 task.

            The Seq2Seq model can solve the babi:task10k:1 task with the normal
            ParlAI setup, and thus should be able to with a PytorchDataTeacher

            This tests the following setups:
                1. -dt train
                2. -dt train:stream
                3. -dt train:stream:ordered
        """
        dts = [
               'train',
               'train:stream',
               'train:stream:ordered']
        for dt in dts:
            print('Testing test_pyt_train with dt: {}'.format(dt))
            f = io.StringIO()
            with redirect_stdout(f):
                parser = train_setup_args()
                defaults = parser_defaults.copy()
                set_model_file(defaults)
                defaults['datatype'] = dt
                defaults['shuffle'] = True  # for train:stream
                parser.set_defaults(**defaults)
                TrainLoop(parser.parse_args()).train()
            str_output = f.getvalue()
            self.assertTrue(solved_task(str_output),
                            'Teacher could not teach seq2seq with args: '
                            '{}; here is str_output: {}'.format(
                                defaults, str_output
                            ))
        print('\n------Passed `test_pyt_train`------\n')

    def test_pyt_preprocess_train(self):
        """
            Test that the preprocess functionality works with the PytorchDataTeacher
            with a sample TorchAgent (here, the Seq2seq model).

            This tests whether an agent can train to completion with
            these preprocessed examples
        """

        # Second, check that the model will train
        print('Testing test_pyt_preprocess training')
        f = io.StringIO()
        with redirect_stdout(f):
            parser = train_setup_args()
            defaults = parser_defaults.copy()
            set_model_file(defaults)
            defaults['datatype'] = 'train'
            defaults['pytorch_preprocess'] = True
            parser.set_defaults(**defaults)
            TrainLoop(parser.parse_args()).train()

        str_output = f.getvalue()
        self.assertTrue(solved_task(str_output),
                        'Teacher could not teach seq2seq with preprocessed obs, '
                        'output: {}'.format(str_output))
        print('\n------Passed `test_pyt_preprocess_train`------\n')

    def test_pyt_batchsort_train(self):
        """
            Tests the functionality of training with batchsort
            under the following conditions:

            1. -dt train --pytorch_preprocess False
            2. -dt train:stream --pytorch_preprocess False
            3. -dt train --pytorch_preprocess True --batch_sort_field text_vec
        """
        # Next, check that training works
        dt_and_preprocess = [
            ('train', False),
            ('train:stream', False),
            ('train', True)
        ]
        for dt, preprocess in dt_and_preprocess:
            print('Testing test_pyt_batchsort with -dt {} and --preprocess {}'.format(
                dt, preprocess
            ))
            f = io.StringIO()
            with redirect_stdout(f):
                parser = train_setup_args()
                defaults = parser_defaults.copy()
                set_model_file(defaults)
                defaults['datatype'] = dt
                defaults['pytorch_preprocess'] = preprocess
                defaults['pytorch_teacher_batch_sort'] = True
                defaults['batchsize'] = 50
                if preprocess:
                    defaults['batch_sort_field'] = 'text_vec'
                parser.set_defaults(**defaults)
                TrainLoop(parser.parse_args()).train()

            str_output = f.getvalue()
            self.assertTrue(solved_task(str_output),
                            'Teacher could not teach seq2seq with batch sort '
                            'and args {} and output {}'.format((dt, preprocess),
                                                               str_output))
        print('\n------Passed `test_pyt_batchsort_train`------\n')

    def test_pytd_teacher(self):
        """
            Test that the pytorch teacher works with given Pytorch Datasets
            as well

            I'll be using the Flickr30k dataset to ensure that the observations
            are the same.
        """
        defaults = parser_defaults.copy()
        defaults['datatype'] = 'train:stream'
        defaults['image_mode'] = 'ascii'

        f = io.StringIO()

        with redirect_stdout(f):
            # Get processed act from agent
            parser = display_setup_args()
            defaults['pytorch_teacher_dataset'] = 'flickr30k'
            del defaults['pytorch_teacher_task']
            parser.set_defaults(**defaults)
            opt = parser.parse_args()
            teacher = create_task_agent_from_taskname(opt)[0]
            pytorch_teacher_act = teacher.act()

            parser = display_setup_args()
            defaults['task'] = 'flickr30k'
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
                            'as regular teacher for act key: {}'.format(key))
        print('\n------Passed `test_pytd_teacher`------\n')

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
            f = io.StringIO()
            with redirect_stdout(f):
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

        task1 = 'babi:task1k:1'
        task2 = 'babi:task1k:2'
        dataset1 = 'flickr30k'
        dataset2 = 'vqa_v1'

        # Expected example and episode counts
        eps_and_exs_counts = [
            (1800, 1800),
            (1080, 1800),
            (29900, 29900),
            (29180, 29900),
            (277349, 277349)
        ]
        defaults = parser_defaults.copy()

        # 1.
        print('Multitask Test 1')
        defaults['pytorch_teacher_task'] = '{},{}'.format(task1, task2)
        run_display_test(defaults, eps_and_exs_counts[0])

        # 2.
        print('Multitask Test 2')
        defaults['pytorch_teacher_task'] = task1
        defaults['task'] = task2
        run_display_test(defaults, eps_and_exs_counts[1])

        # 3.
        print('Multitask Test 3')
        del defaults['task']
        defaults['pytorch_teacher_dataset'] = dataset1
        run_display_test(defaults, eps_and_exs_counts[2])

        # 4.
        print('Multitask Test 4')
        del defaults['pytorch_teacher_task']
        defaults['task'] = task1
        run_display_test(defaults, eps_and_exs_counts[3])

        # 5.
        print('Multitask Test 5')
        del defaults['task']
        defaults['pytorch_teacher_dataset'] = '{},{}'.format(dataset1, dataset2)
        run_display_test(defaults, eps_and_exs_counts[4])
        print('\n------Passed `test_pyt_multitask`------\n')


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.scripts.build_dict import build_dict
from parlai.scripts.display_data import setup_args as display_setup_args, \
    display_data
from parlai.scripts.train_model import TrainLoop, setup_args as train_setup_args
from parlai.core.agents import create_task_agent_from_taskname, create_agent
from parlai.core.worlds import create_task
from parlai.core.pytorch_data_teacher import ep_length

import unittest
import tempfile
import io
import os
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler as Sequential
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
    """Various tests for PytorchDataTeacher"""

    def test_shuffle(self):
        """Simple test to ensure that dataloader is initialized with correct
            data sampler
        """
        dts = ['train', 'valid', 'test']
        exts = ['', ':stream', ':ordered', ':stream:ordered']
        shuffle_opts = [False, True]
        task = 'babi:task1k:1'
        for dt in dts:
            for ext in exts:
                datatype = dt + ext
                for shuffle in shuffle_opts:
                    opt_defaults = {
                        'pytorch_teacher_task': task,
                        'datatype': datatype,
                        'shuffle': shuffle
                    }
                    print('Testing test_shuffle with args {}'.format(opt_defaults))
                    f = io.StringIO()
                    with redirect_stdout(f):
                        parser = display_setup_args()
                        parser.set_defaults(**opt_defaults)
                        opt = parser.parse_args()
                        teacher = create_task_agent_from_taskname(opt)[0]
                    if ('ordered' in datatype or
                            ('stream' in datatype and not opt.get('shuffle')) or
                            'train' not in datatype):
                        self.assertTrue(
                            type(teacher.pytorch_dataloader.sampler) is Sequential,
                            'PytorchDataTeacher failed with args: {}'.format(opt)
                        )
                    else:
                        self.assertTrue(
                            type(teacher.pytorch_dataloader.sampler) is RandomSampler,
                            'PytorchDataTeacher failed with args: {}'.format(opt)
                        )
        print('\n------Passed `test_shuffle`------\n')

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

    def test_pyt_preprocess(self):
        """
            Test that the preprocess functionality works with the PytorchDataTeacher
            with a sample TorchAgent (here, the Seq2seq model).

            This tests whether the action provided by the preprocessed teacher
            is equivalent to the agent's observation after the agent processes it.

        """
        def get_teacher_act(defaults, teacher_processed=False, agent_to=None):

            parser = train_setup_args()
            parser.set_defaults(**defaults)
            opt = parser.parse_args()
            build_dict(opt)
            teacher = create_task_agent_from_taskname(opt)[0]
            agent = create_agent(opt)
            act = teacher.act()
            if teacher_processed:
                return act, agent
            return agent.observe(act), agent

        print('Testing test_pyt_preprocess action equivalent to observation')
        ff = io.StringIO()
        with redirect_stdout(ff):
            defaults = parser_defaults.copy()
            defaults['batch_size'] = 1
            defaults['datatype'] = 'train:stream:ordered'

            # Get processed act from agent
            set_model_file(defaults)
            agent_processed_observation, agent1 = get_teacher_act(defaults)

            # Get preprocessed act from teacher
            set_model_file(defaults)
            defaults['pytorch_preprocess'] = True
            teacher_processed_act, agent2 = get_teacher_act(defaults,
                                                            teacher_processed=True)

        for key in agent_processed_observation:
            val1 = agent_processed_observation[key]
            val2 = teacher_processed_act[key]
            if type(val1) is torch.Tensor:
                self.assertTrue(bool(torch.all(torch.eq(val1, val2))))
            else:
                self.assertTrue(val1 == val2,
                                '{}\n\n --not equal to-- \n\n{}'.format(
                                    val1,
                                    val2)
                                )
        print('\n------Passed `test_pyt_preprocess`------\n')

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
                        'Teacher could not teach seq2seq with preprocessed obs')
        print('\n------Passed `test_pyt_preprocess_train`------\n')

    def test_pyt_batchsort(self):
        """
            Tests that batchsort *works* for two epochs; that is, that
            every example is seen both epochs
        """
        parser = train_setup_args()

        def get_acts_epochs_1_and_2(defaults):
            parser.set_defaults(**defaults)
            opt = parser.parse_args()
            build_dict(opt)
            agent = create_agent(opt)
            world_data = create_task(opt, agent)
            acts_epoch_1 = []
            acts_epoch_2 = []
            while not world_data.epoch_done():
                world_data.parley()
                acts_epoch_1.append(world_data.acts[0])
            world_data.reset()
            while not world_data.epoch_done():
                world_data.parley()
                acts_epoch_2.append(world_data.acts[0])
            acts_epoch_1 = [bb for b in acts_epoch_1 for bb in b]
            acts_epoch_1 = sorted([b for b in acts_epoch_1 if 'text' in b],
                                  key=lambda x: x.get('text'))
            acts_epoch_2 = [bb for b in acts_epoch_2 for bb in b]
            acts_epoch_2 = sorted([b for b in acts_epoch_2 if 'text' in b],
                                  key=lambda x: x.get('text'))
            return acts_epoch_1, acts_epoch_2

        def check_equal_act_lists(acts1, acts2):
            for idx in range(len(acts1)):
                act1 = acts1[idx]
                act2 = acts2[idx]
                for key in act1:
                    val1 = act1[key]
                    val2 = act2[key]
                    if type(val1) is torch.Tensor:
                        self.assertTrue(bool(torch.all(torch.eq(val1, val2))))
                    else:
                        self.assertTrue(val1 == val2,
                                        '{}\n\n --not equal to-- \n\n{}'.format(
                                            val1,
                                            val2)
                                        )
        # First, check that batchsort itself works
        defaults = parser_defaults.copy()
        defaults['datatype'] = 'train:stream:ordered'
        defaults['pytorch_teacher_batch_sort'] = True

        f = io.StringIO()

        with redirect_stdout(f):
            # Get processed act from agent
            defaults['pytorch_teacher_task'] = 'babi:task1k:1'
            defaults['batch_sort_cache_type'] = 'index'
            defaults['batchsize'] = 50
            set_model_file(defaults)
            bsrt_acts_ep1, bsrt_acts_ep2 = get_acts_epochs_1_and_2(defaults)

            defaults['pytorch_teacher_batch_sort'] = False
            set_model_file(defaults)
            no_bsrt_acts_ep1, no_bsrt_acts_ep2 = get_acts_epochs_1_and_2(defaults)

        check_equal_act_lists(bsrt_acts_ep1, no_bsrt_acts_ep1)
        check_equal_act_lists(bsrt_acts_ep2, no_bsrt_acts_ep2)
        print('\n------Passed `test_pyt_batchsort`------\n')

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

    def test_pyt_batchsort_field(self):
        """
            Test that the batchsort actually works for Pytorch Data Teacher

            That is, for every batch except the last one, check that the length
            of each example (determined by batchsort_field) is within
            `batch-length-range`

        """
        # First, check that batchsort itself works
        defaults = parser_defaults.copy()
        defaults['datatype'] = 'train:stream:ordered'
        defaults['pytorch_teacher_batch_sort'] = True
        defaults['pytorch_teacher_task'] = 'babi:task1k:1'
        defaults['batchsize'] = 50
        max_range = defaults['batch_length_range']

        def verify_batch_lengths(defaults):
            f = io.StringIO()

            with redirect_stdout(f):
                # Get processed act from agent
                parser = train_setup_args()
                set_model_file(defaults)
                parser.set_defaults(**defaults)
                opt = parser.parse_args()
                build_dict(opt)
                agent = create_agent(opt)
                world_data = create_task(opt, agent)
                batch_sort_acts = []
                # first epoch
                while len(batch_sort_acts) < 900/50:
                    world_data.parley()
                    batch_sort_acts.append(world_data.acts[0])
                teacher = world_data.world.get_agents()[0]
                teacher.reset_data()
                # second epoch
                while len(batch_sort_acts) < 1800/50:
                    world_data.parley()
                    batch_sort_acts.append(world_data.acts[0])
            field = defaults['batch_sort_field']
            lengths = [[ep_length(b[field]) for b in bb if field in b]
                       for bb in batch_sort_acts[:-2]]  # exclude last batch
            # verify batch lengths
            for batch_lens in lengths:
                self.assertLessEqual(max(batch_lens) - min(batch_lens), max_range,
                                     'PytorchDataTeacher batching does not give '
                                     'batches with similar sized examples, when '
                                     'sorting by `{}` field.'.format(
                                        defaults['batch_sort_field']))
        defaults['batch_sort_field'] = 'text'
        verify_batch_lengths(defaults)
        defaults['batch_sort_field'] = 'text_vec'
        defaults['pytorch_preprocess'] = True
        verify_batch_lengths(defaults)
        print('\n------Passed `test_pyt_batchsort_field`------\n')

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

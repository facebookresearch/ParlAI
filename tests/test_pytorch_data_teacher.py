#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.scripts.build_dict import build_dict
from parlai.scripts.display_data import setup_args as display_setup_args
from parlai.scripts.train_model import setup_args as train_setup_args
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
    """Various Unit tests for PytorchDataTeacher"""

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
                self.assertTrue(bool(torch.all(torch.eq(val1, val2))),
                                '{}\n\n --not equal to-- \n\n{}'.format(
                                    val1,
                                    val2)
                                )
            else:
                self.assertTrue(val1 == val2,
                                '{}\n\n --not equal to-- \n\n{}'.format(
                                    val1,
                                    val2)
                                )
        print('\n------Passed `test_pyt_preprocess`------\n')

    def test_valid_pyt_batchsort(self):
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


if __name__ == '__main__':
    unittest.main()

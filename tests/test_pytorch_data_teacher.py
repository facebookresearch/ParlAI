#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.scripts.build_dict import build_dict
from parlai.scripts.display_data import setup_args as display_setup_args, display_data
from parlai.scripts.train_model import setup_args as train_setup_args
from parlai.core.agents import create_task_agent_from_taskname, create_agent
from parlai.core.worlds import create_task
from parlai.core.pytorch_data_teacher import ep_length

import unittest
import parlai.core.testing_utils as testing_utils
import os
import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler as Sequential

unit_test_parser_defaults = {
    'model': 'seq2seq',
    'pytorch_teacher_task': 'babi:task10k:1',
    'model_file': '/tmp/tmp_model',
    'batchsize': 32,
    'momentum':  0.9,
    'validation_every_n_secs': 30,
    'batch_length_range': 5,
}


integration_test_parser_defaults = {
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
    """Various tests for PytorchDataTeacher"""

    """Unit Tests"""

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
                    with testing_utils.capture_output() as _:
                        parser = display_setup_args()
                        parser.set_defaults(**opt_defaults)
                        opt = parser.parse_args()
                        teacher = create_task_agent_from_taskname(opt)[0]
                        if (
                            'ordered' in datatype or
                            ('stream' in datatype and not opt.get('shuffle')) or
                            'train' not in datatype
                        ):
                            self.assertIsInstance(
                                teacher.pytorch_dataloader.sampler, Sequential,
                                'PytorchDataTeacher failed with args: {}'.format(opt)
                            )
                        else:
                            self.assertIsInstance(
                                teacher.pytorch_dataloader.sampler, RandomSampler,
                                'PytorchDataTeacher failed with args: {}'.format(opt)
                            )

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
            with testing_utils.capture_output() as _:
                teacher = create_task_agent_from_taskname(opt)[0]
            agent = create_agent(opt)
            act = teacher.act()
            if teacher_processed:
                return act, agent
            return agent.observe(act), agent

        with testing_utils.capture_output() as _, testing_utils.tempdir() as tmpdir:
            defaults = unit_test_parser_defaults.copy()
            defaults['batch_size'] = 1
            defaults['datatype'] = 'train:stream:ordered'

            # Get processed act from agent
            defaults['model_file'] = os.path.join(tmpdir, 'model')
            defaults['dict_file'] = os.path.join(tmpdir, 'model.dict')
            agent_processed_observation, agent1 = get_teacher_act(defaults)

            # Get preprocessed act from teacher
            defaults['model_file'] = os.path.join(tmpdir, 'model')
            defaults['dict_file'] = os.path.join(tmpdir, 'model.dict')
            defaults['pytorch_preprocess'] = True
            teacher_processed_act, agent2 = get_teacher_act(defaults, teacher_processed=True)  # noqa: E501

        for key in agent_processed_observation:
            val1 = agent_processed_observation[key]
            val2 = teacher_processed_act[key]
            if isinstance(val1, torch.Tensor):
                self.assertTrue(
                    bool(torch.all(torch.eq(val1, val2))),
                    '{} is not equal to {}'.format(val1, val2)
                )
            else:
                self.assertEqual(val1, val2)

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
            world_data.shutdown()
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
                        self.assertEqual(val1, val2)
        # First, check that batchsort itself works
        defaults = unit_test_parser_defaults.copy()
        defaults['datatype'] = 'train:stream:ordered'
        defaults['pytorch_teacher_batch_sort'] = True

        with testing_utils.capture_output() as _, testing_utils.tempdir() as tmpdir:
            # Get processed act from agent
            defaults['pytorch_teacher_task'] = 'babi:task1k:1'
            defaults['batch_sort_cache_type'] = 'index'
            defaults['batchsize'] = 50
            defaults['model_file'] = os.path.join(tmpdir, 'model')
            defaults['dict_file'] = os.path.join(tmpdir, 'model.dict')
            bsrt_acts_ep1, bsrt_acts_ep2 = get_acts_epochs_1_and_2(defaults)

            defaults['pytorch_teacher_batch_sort'] = False
            defaults['model_file'] = os.path.join(tmpdir, 'model')
            defaults['dict_file'] = os.path.join(tmpdir, 'model.dict')
            no_bsrt_acts_ep1, no_bsrt_acts_ep2 = get_acts_epochs_1_and_2(defaults)

        check_equal_act_lists(bsrt_acts_ep1, no_bsrt_acts_ep1)
        check_equal_act_lists(bsrt_acts_ep2, no_bsrt_acts_ep2)

    def test_pyt_batchsort_field(self):
        """
        Test that the batchsort actually works for Pytorch Data Teacher

        That is, for every batch except the last one, check that the length
        of each example (determined by batchsort_field) is within
        `batch-length-range`
        """
        # First, check that batchsort itself works
        defaults = unit_test_parser_defaults.copy()
        defaults['datatype'] = 'train:stream:ordered'
        defaults['pytorch_teacher_batch_sort'] = True
        defaults['pytorch_teacher_task'] = 'babi:task1k:1'
        defaults['batchsize'] = 50
        max_range = defaults['batch_length_range']

        def verify_batch_lengths(defaults):
            with testing_utils.capture_output() as _, testing_utils.tempdir() as tmpdir:
                # Get processed act from agent
                parser = train_setup_args()
                defaults['model_file'] = os.path.join(tmpdir, 'model')
                defaults['dict_file'] = os.path.join(tmpdir, 'model.dict')
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
                world_data.shutdown()
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

    """Integration Tests"""

    def _pyt_train(self, datatype):
        """
        Integration test: ensure that pytorch data teacher can successfully
        teach Seq2Seq model to fully solve the babi:task10k:1 task.

        The Seq2Seq model can solve the babi:task10k:1 task with the normal
        ParlAI setup, and thus should be able to with a PytorchDataTeacher
        """
        defaults = integration_test_parser_defaults.copy()
        defaults['datatype'] = datatype
        defaults['shuffle'] = True  # for train:stream
        str_output, valid, test = testing_utils.train_model(defaults)
        self.assertTrue(
            solved_task(str_output, valid, test),
            'Teacher could not teach seq2seq with args: {}; here is str_output: {}'
            .format(defaults, str_output)
        )

    @testing_utils.retry()
    def test_pyt_train(self):
        self._pyt_train('train')

    @testing_utils.retry()
    def test_pyt_train_stream(self):
        self._pyt_train('train:stream')

    @testing_utils.retry()
    def test_pyt_train_stream_ordered(self):
        self._pyt_train('train:stream:ordered')

    @testing_utils.retry()
    def test_pyt_preprocess_train(self):
        """
        Test that the preprocess functionality works with the PytorchDataTeacher
        with a sample TorchAgent (here, the Seq2seq model).

        This tests whether an agent can train to completion with
        these preprocessed examples
        """
        defaults = integration_test_parser_defaults.copy()
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

        :param string datatype:
            datatype to train with
        :param bool preprocess:
            whether to preprocess the data
        """
        defaults = integration_test_parser_defaults.copy()
        defaults['datatype'] = datatype
        defaults['pytorch_preprocess'] = preprocess
        defaults['pytorch_teacher_batch_sort'] = True
        if preprocess:
            defaults['batch_sort_field'] = 'text_vec'
        str_output, valid, test = testing_utils.train_model(defaults)
        self.assertTrue(
            solved_task(str_output, valid, test),
            'Teacher could not teach seq2seq with batch sort '
            'and args {} and output {}'
            .format((datatype, preprocess), str_output)
        )

    @testing_utils.retry()
    def test_pyt_batchsort_train(self):
        self._pyt_batchsort_train('train', False)

    @testing_utils.retry()
    def test_pyt_batchsort_train_stream(self):
        self._pyt_batchsort_train('train:stream', False)

    @testing_utils.retry()
    def test_pyt_batchsort_train_preprocess(self):
        self._pyt_batchsort_train('train', True)

    def test_pytd_teacher(self):
        """
        Test that the pytorch teacher works with given Pytorch Datasets
        as well
        """
        defaults = integration_test_parser_defaults.copy()
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
        defaults = integration_test_parser_defaults.copy()

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

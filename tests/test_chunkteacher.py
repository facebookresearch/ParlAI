#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test correctness of ChunkTeacher in a large number of settings.
"""

from unittest import TestCase
import os
import parlai.utils.testing as testing_utils
from parlai.tasks.integration_tests.agents import NUM_TEST
import parlai.scripts.multiprocessing_train as mp_train


class _Abstract(TestCase):
    BASE_ARGS = {
        'model': 'test_agents/counter',
        'dict_file': 'zoo:unittest/transformer_generator2/model.dict',
        'dict_tokenizer': 'space',
        'truncate': 8,
        'max_train_steps': 10,
        'datatype': 'train:stream',
    }

    TASK = None

    def _run(self, **kwargs):
        opt = {**self.BASE_ARGS, **kwargs}
        opt['task'] = self.TASK
        valid_report, test_report = testing_utils.train_model(opt)
        assert valid_report['unique'] == NUM_TEST
        assert valid_report['times_seen'] == 1
        assert test_report['unique'] == NUM_TEST
        assert test_report['times_seen'] == 1
        return valid_report, test_report

    def _run_mp(self, **kwargs):
        opt = {**self.BASE_ARGS, **kwargs}
        opt['task'] = self.TASK
        with testing_utils.tempdir() as tmpdir:
            if 'model_file' not in opt:
                opt['model_file'] = os.path.join(tmpdir, 'model')

            valid_report, test_report = mp_train.MultiProcessTrain.main(**opt)
            assert valid_report['unique'] == NUM_TEST
            assert valid_report['times_seen'] == 1
            assert test_report['unique'] == NUM_TEST
            assert test_report['times_seen'] == 1
            return valid_report, test_report


class TestNumExamples(_Abstract):
    TASK = 'integration_tests:chunky'

    # Regular chunk teacher
    def test_normal_bs1(self):
        self._run()

    def test_normal_bs3(self):
        self._run(batchsize=3)

    def test_normal_dynb(self):
        self._run(batchsize=2, dynamic_batching='full')

    def test_normal_batchsort(self):
        self._run(batchsize=2, dynamic_batching='batchsort')

    @testing_utils.skipUnlessGPU
    def test_mp_normal_bs1(self):
        self._run_mp(task='integration_tests:chunky', batchsize=1)

    @testing_utils.skipUnlessGPU
    def test_mp_normal_bs3(self):
        self._run_mp(task='integration_tests:chunky', batchsize=3)

    @testing_utils.skipUnlessGPU
    def test_mp_normal_dynb(self):
        self._run_mp(
            task='integration_tests:chunky', batchsize=2, dynamic_batching='full'
        )


class TestSmallBuffer(_Abstract):
    # Small buffer
    TASK = 'integration_tests:chunky_small_buffer'

    def test_small_buffer_bs1(self):
        self._run()

    def test_small_buffer_bs3(self):
        self._run(batchsize=3)

    def test_small_buffer_dynb(self):
        self._run(batchsize=2, dynamic_batching='full')

    def test_small_buffer_batchsort(self):
        self._run(batchsize=2, dynamic_batching='batchsort')

    @testing_utils.skipUnlessGPU
    def test_mp_small_buffer_bs1(self):
        self._run_mp()

    @testing_utils.skipUnlessGPU
    def test_mp_small_buffer_bs3(self):
        self._run_mp(batchsize=3)

    @testing_utils.skipUnlessGPU
    def test_mp_small_buffer_dynb(self):
        self._run_mp(batchsize=2, dynamic_batching='full')

    @testing_utils.skipUnlessGPU
    def test_mp_small_buffer_batchsort(self):
        self._run_mp(batchsize=2, dynamic_batching='batchsort')


class TestSlowChunk(_Abstract):
    TASK = 'integration_tests:chunky_slow'

    # Slow chunk
    def test_slow_bs3(self):
        self._run(batchsize=3)

    def test_slow_dynb(self):
        self._run(batchsize=2, dynamic_batching='full')


class TestBackgroundPreprocessorNumExamples(TestNumExamples):
    BASE_ARGS = {
        'model': 'test_agents/counter',
        'dict_file': 'zoo:unittest/transformer_generator2/model.dict',
        'dict_tokenizer': 'space',
        'truncate': 8,
        'max_train_steps': 10,
        'datatype': 'train:stream',
        'num_workers': 4,
    }


class TestWrongExamples(TestNumExamples):
    TASK = 'integration_tests:wrong_examples_chunky'


class TestWrongEpisodes(TestNumExamples):
    TASK = 'integration_tests:wrong_episodes_chunky'


class TestWrongExamplesEpisodes(TestNumExamples):
    TASK = 'integration_tests:wrong_examples_episodes_chunky'

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
import torch.distributed as dist
import parlai.scripts.multiprocessing_train as mp_train

BASE_ARGS = {
    'model': 'test_agents/counter',
    'dict_file': 'zoo:unittest/transformer_generator2/model.dict',
    'dict_tokenizer': 'space',
    'truncate': 8,
    'num_epochs': 2,
    'datatype': 'train:stream',
}


class TestNumExamples(TestCase):
    def _run(self, **kwargs):
        opt = {**BASE_ARGS, **kwargs}
        valid_report, test_report = testing_utils.train_model(opt)
        assert valid_report['unique'] == NUM_TEST
        assert valid_report['times_seen'] == 1
        assert test_report['unique'] == NUM_TEST
        assert test_report['times_seen'] == 1
        return valid_report, test_report

    def _run_mp(self, **kwargs):
        opt = {**BASE_ARGS, **kwargs}
        with testing_utils.tempdir() as tmpdir:
            if 'model_file' not in opt:
                opt['model_file'] = os.path.join(tmpdir, 'model')

            valid_report, test_report = mp_train.MultiProcessTrain.main(**opt)
            dist.destroy_process_group()
            assert valid_report['unique'] == NUM_TEST
            assert valid_report['times_seen'] == 1
            assert test_report['unique'] == NUM_TEST
            assert test_report['times_seen'] == 1
            return valid_report, test_report

    # Regular chunk teacher

    def test_normal_bs1(self):
        self._run(task='integration_tests:chunky')

    def test_normal_bs2(self):
        self._run(task='integration_tests:chunky', batchsize=2)

    def test_normal_bs3(self):
        self._run(task='integration_tests:chunky', batchsize=3)

    def test_normal_dynb(self):
        self._run(task='integration_tests:chunky', batchsize=2, dynamic_batching='full')

    def test_normal_batchsort(self):
        self._run(
            task='integration_tests:chunky', batchsize=2, dynamic_batching='batchsort'
        )

    def test_mp_normal_bs1(self):
        self._run_mp(task='integration_tests:chunky')

    def test_mp_normal_bs2(self):
        self._run_mp(task='integration_tests:chunky', batchsize=2)

    def test_mp_normal_bs3(self):
        self._run_mp(task='integration_tests:chunky', batchsize=3)

    def test_mp_normal_dynb(self):
        self._run_mp(
            task='integration_tests:chunky', batchsize=2, dynamic_batching='full'
        )

    def test_mp_normal_batchsort(self):
        self._run_mp(
            task='integration_tests:chunky', batchsize=2, dynamic_batching='batchsort'
        )

    # Small buffer

    def test_small_buffer_bs1(self):
        self._run(task='integration_tests:chunky_small_buffer')

    def test_small_buffer_bs2(self):
        self._run(task='integration_tests:chunky_small_buffer', batchsize=2)

    def test_small_buffer_bs3(self):
        self._run(task='integration_tests:chunky_small_buffer', batchsize=3)

    def test_small_buffer_dynb(self):
        self._run(
            task='integration_tests:chunky_small_buffer',
            batchsize=2,
            dynamic_batching='full',
        )

    def test_small_buffer_batchsort(self):
        self._run(
            task='integration_tests:chunky_small_buffer',
            batchsize=2,
            dynamic_batching='batchsort',
        )

    def test_mp_small_buffer_bs1(self):
        self._run_mp(task='integration_tests:chunky_small_buffer')

    def test_mp_small_buffer_bs2(self):
        self._run_mp(task='integration_tests:chunky_small_buffer', batchsize=2)

    def test_mp_small_buffer_bs3(self):
        self._run_mp(task='integration_tests:chunky_small_buffer', batchsize=3)

    def test_mp_small_buffer_dynb(self):
        self._run_mp(
            task='integration_tests:chunky_small_buffer',
            batchsize=2,
            dynamic_batching='full',
        )

    def test_mp_small_buffer_batchsort(self):
        self._run_mp(
            task='integration_tests:chunky_small_buffer',
            batchsize=2,
            dynamic_batching='batchsort',
        )

    # Slow chunk

    def test_slow_bs1(self):
        self._run(task='integration_tests:chunky_slow')

    def test_slow_bs2(self):
        self._run(task='integration_tests:chunky_slow', batchsize=2)

    def test_slow_bs3(self):
        self._run(task='integration_tests:chunky_slow', batchsize=3)

    def test_slow_dynb(self):
        self._run(
            task='integration_tests:chunky_slow', batchsize=2, dynamic_batching='full'
        )

    def test_slow_batchsort(self):
        self._run(
            task='integration_tests:chunky_slow',
            batchsize=2,
            dynamic_batching='batchsort',
        )

    def test_mp_slow_bs1(self):
        self._run_mp(task='integration_tests:chunky_slow')

    def test_mp_slow_bs2(self):
        self._run_mp(task='integration_tests:chunky_slow', batchsize=2)

    def test_mp_slow_bs3(self):
        self._run_mp(task='integration_tests:chunky_slow', batchsize=3)

    def test_mp_slow_dynb(self):
        self._run_mp(
            task='integration_tests:chunky_slow', batchsize=2, dynamic_batching='full'
        )

    def test_mp_slow_batchsort(self):
        self._run_mp(
            task='integration_tests:chunky_slow',
            batchsize=2,
            dynamic_batching='batchsort',
        )

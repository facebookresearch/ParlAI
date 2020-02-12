#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Any

import unittest
from parlai.core.opt import Opt
import parlai.utils.testing as testing_utils
from parlai.tasks.integration_tests.agents import NUM_TEST, EXAMPLE_SIZE

_TASK = 'integration_tests:variable_length'
_DEFAULT_OPTIONS = {
    'batchsize': 8,
    'dynamic_batching': 'full',
    'optimizer': 'adamax',
    'learningrate': 7e-3,
    'num_epochs': 1,
    'n_layers': 1,
    'n_heads': 1,
    'ffn_size': 32,
    'embedding_size': 32,
    'truncate': 8,
    'model': 'transformer/generator',
    'task': _TASK,
}


# TODO tests to write:
# - multiple validation runs, streaming/not streaming
# - ranking model


class TestDynamicBatching(unittest.TestCase):
    def _test_correct_processed(self, num_goal: int, **kwargs: Dict[str, Any]):
        opt = Opt({**_DEFAULT_OPTIONS, **kwargs})
        valid_report, test_report = testing_utils.train_model(opt)
        self.assertEqual(valid_report['exs'], num_goal)
        self.assertEqual(test_report['exs'], num_goal)

    def test_no_truncate(self):
        with self.assertRaises(ValueError):
            testing_utils.train_model(Opt({**_DEFAULT_OPTIONS, **{'truncate': -1}}))

    def test_no_batch_act(self):
        """
        Fail when the agent doesn't support dynamic batching.
        """
        with self.assertRaises(TypeError):
            testing_utils.train_model(model='repeat_label', task=_TASK)

        with self.assertRaises(TypeError):
            testing_utils.eval_model(model='repeat_label', task=_TASK)

    def test_ranking(self):
        self._test_correct_processed(
            NUM_TEST, model='transformer/ranker', datatype='train'
        )

    def test_ranking_streaming(self):
        self._test_correct_processed(
            NUM_TEST, model='transformer/ranker', datatype='train:stream'
        )

    def test_training(self):
        self._test_correct_processed(NUM_TEST, datatype='train')

    def test_streaming(self):
        self._test_correct_processed(NUM_TEST, datatype='train:stream')

    def test_multiworld(self):
        self._test_correct_processed(
            NUM_TEST + NUM_TEST * EXAMPLE_SIZE,
            task='integration_tests:variable_length,integration_tests:multiturn',
        )

    def test_multiworld_stream(self):
        self._test_correct_processed(
            NUM_TEST + NUM_TEST * EXAMPLE_SIZE,
            task='integration_tests:variable_length,integration_tests:multiturn',
            datatype='train:stream',
        )


class TestBatchSort(unittest.TestCase):
    def _test_correct_processed(self, num_goal: int, **kwargs: Dict[str, Any]):
        opt = Opt({**_DEFAULT_OPTIONS, **kwargs})
        opt['dynamic_batching'] = 'batchsort'
        valid_report, test_report = testing_utils.train_model(opt)
        self.assertEqual(valid_report['exs'], num_goal)
        self.assertEqual(test_report['exs'], num_goal)

    def test_no_batch_act(self):
        """
        Fail when the agent doesn't support dynamic batching.
        """
        with self.assertRaises(TypeError):
            testing_utils.train_model(model='repeat_label', task=_TASK)

        with self.assertRaises(TypeError):
            testing_utils.eval_model(model='repeat_label', task=_TASK)

    def test_ranking(self):
        self._test_correct_processed(
            NUM_TEST, model='transformer/ranker', datatype='train'
        )

    def test_ranking_streaming(self):
        self._test_correct_processed(
            NUM_TEST, model='transformer/ranker', datatype='train:stream'
        )

    def test_training(self):
        self._test_correct_processed(NUM_TEST, datatype='train')

    def test_streaming(self):
        self._test_correct_processed(NUM_TEST, datatype='train:stream')

    def test_multiworld(self):
        self._test_correct_processed(
            NUM_TEST + NUM_TEST * EXAMPLE_SIZE,
            task='integration_tests:variable_length,integration_tests:multiturn',
        )

    def test_multiworld_stream(self):
        self._test_correct_processed(
            NUM_TEST + NUM_TEST * EXAMPLE_SIZE,
            task='integration_tests:variable_length,integration_tests:multiturn',
            datatype='train:stream',
        )


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

from typing import Dict, Any

import unittest
from parlai.core.opt import Opt
import parlai.utils.testing as testing_utils

_TASK = 'integration_tests:variable_length'
_DEFAULT_OPTIONS = {
    'batchsize': 8,
    'dynamic_batching': True,
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


class TestDynamicBatching(unittest.TestCase):
    def _test_correct_processed(self, **kwargs: Dict[str, Any]):
        opt = Opt({**_DEFAULT_OPTIONS, **kwargs})
        train_log, valid_report, test_report = testing_utils.train_model(opt)
        self.assertEqual(valid_report['exs'], 100)
        self.assertEqual(test_report['exs'], 100)

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

    def test_training(self):
        self._test_correct_processed(datatype='train')

    def test_streaming(self):
        self._test_correct_processed(datatype='train:stream')

    def test_multiworld(self):
        self._test_correct_processed(
            task='integration_tests:variable_length,integration_tests',
        )

    def test_multiworld_stream(self):
        self._test_correct_processed(
            task='integration_tests:variable_length,integration_tests',
            datatype='train:stream',
        )

    # tests to write:
    # - multiple validation runs, streaming/not streaming
    # - ranking model


if __name__ == '__main__':
    unittest.main()

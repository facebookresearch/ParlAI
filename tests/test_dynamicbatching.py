#!/usr/bin/env python3

import unittest
from parlai.core.opt import Opt
import parlai.utils.testing as testing_utils

_BASE_OPTIONS = {
    'batchsize': 8,
    'dynamic_batching': True,
    'task': 'integration_tests:variable_length',
}


class TestDynamicBatching(unittest.TestCase):
    def test_no_truncate(self):
        with self.assertRaises(ValueError):
            testing_utils.train_model(
                Opt({'model': 'transformer/generator', **_BASE_OPTIONS})
            )

    def test_no_batch_act(self):
        """
        Fail when the agent doesn't support dynamic batching.
        """
        with self.assertRaises(TypeError):
            testing_utils.train_model(
                Opt({'model': 'repeat_label', 'truncate': 8, **_BASE_OPTIONS})
            )

        with self.assertRaises(TypeError):
            testing_utils.eval_model(
                Opt({'model': 'repeat_label', 'text_truncate': 8, **_BASE_OPTIONS})
            )

    def test_training(self):
        testing_utils.train_model(
            dict(
                model='transformer/generator',
                optimizer='adamax',
                learningrate=7e-3,
                num_epochs=1,
                n_layers=1,
                n_heads=1,
                ffn_size=32,
                embedding_size=32,
                inference='beam',
                truncate=8,
                **_BASE_OPTIONS,
            )
        )

    def test_multiworld(self):
        testing_utils.train_model(
            Opt(
                {
                    'model': 'transformers/generator',
                    'task': 'integration_tests:variable_length,integration_tests',
                    'truncate': 8,
                    **_BASE_OPTIONS,
                }
            )
        )


if __name__ == '__main__':
    unittest.main()

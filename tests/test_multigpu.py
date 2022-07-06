#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils

MODEL_OPTS = {
    'n_layers': 4,
    'embedding_size': 16,
    'ffn_size': 32,
    'n_heads': 2,
    'num_epochs': 0.1,
    'batchsize': 32,
    'truncate': 8,
}


@testing_utils.skipUnlessGPU
class TestModelParallel(unittest.TestCase):
    def test_polyencoder(self):
        testing_utils.train_model(
            {
                'task': 'integration_tests',
                'model': 'transformer/polyencoder',
                'model_parallel': True,
                'candidates': 'batch',
                'poly_n_codes': 4,
                **MODEL_OPTS,
            }
        )

        with self.assertRaises(RuntimeError):
            testing_utils.train_model(
                {
                    'task': 'integration_tests',
                    'model': 'transformer/polyencoder',
                    'data_parallel': True,
                    'model_parallel': True,
                    'candidates': 'batch',
                    'poly_n_codes': 4,
                    **MODEL_OPTS,
                }
            )

    def test_ranker(self):
        testing_utils.train_model(
            {
                'task': 'integration_tests',
                'model': 'transformer/ranker',
                'candidates': 'batch',
                'model_parallel': True,
                **MODEL_OPTS,
            }
        )

        with self.assertRaises(RuntimeError):
            testing_utils.train_model(
                {
                    'task': 'integration_tests',
                    'model': 'transformer/ranker',
                    'data_parallel': True,
                    'model_parallel': True,
                    'candidates': 'batch',
                    **MODEL_OPTS,
                }
            )

    def test_classifier(self):
        testing_utils.train_model(
            {
                'task': 'integration_tests:classifier',
                'classes': ['one', 'zero'],
                'model': 'transformer/classifier',
                'model_parallel': True,
                **MODEL_OPTS,
            }
        )
        with self.assertRaises(RuntimeError):
            testing_utils.train_model(
                {
                    'task': 'integration_tests:classifier',
                    'classes': ['one', 'zero'],
                    'model': 'transformer/classifier',
                    'data_parallel': True,
                    'model_parallel': True,
                    **MODEL_OPTS,
                }
            )

    def test_transformer_generator(self):
        testing_utils.train_model(
            {
                'task': 'integration_tests',
                'model': 'transformer/generator',
                'model_parallel': True,
                **MODEL_OPTS,
            }
        )


@testing_utils.skipUnlessGPU
class TestDataParallel(unittest.TestCase):
    def test_polyencoder(self):
        testing_utils.train_model(
            {
                'task': 'integration_tests',
                'model': 'transformer/polyencoder',
                'candidates': 'batch',
                'poly_n_codes': 4,
                'data_parallel': True,
                **MODEL_OPTS,
            }
        )

    def test_ranker(self):
        testing_utils.train_model(
            {
                'task': 'integration_tests',
                'model': 'transformer/ranker',
                'candidates': 'batch',
                'data_parallel': True,
                **MODEL_OPTS,
            }
        )

    def test_classifier(self):
        testing_utils.train_model(
            {
                'task': 'integration_tests:classifier',
                'classes': ['one', 'zero'],
                'data_parallel': True,
                'model': 'transformer/classifier',
                **MODEL_OPTS,
            }
        )

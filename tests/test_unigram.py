#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


class TestUnigram(unittest.TestCase):
    def test_unigram(self):
        valid, test = testing_utils.train_model(
            {'model': 'unigram', 'task': 'integration_tests', 'num_epochs': 0.01}
        )
        assert valid['f1'] > 0


class TestUnigramTorchAgent(unittest.TestCase):
    def test_unigram(self):
        valid, test = testing_utils.train_model(
            {
                'model': 'test_agents/unigram',
                'task': 'integration_tests',
                'num_epochs': 1.0,
                'batchsize': 32,
                'truncate': 4,
            }
        )
        assert valid['f1'] > 0

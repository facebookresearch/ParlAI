#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


class TestStarspace(unittest.TestCase):
    def test_training(self):
        valid, test = testing_utils.train_model(
            {'model': 'starspace', 'task': 'integration_tests', 'num_epochs': 1.0}
        )

        assert valid['hits@1'] > 0.5
        assert test['hits@1'] > 0.5

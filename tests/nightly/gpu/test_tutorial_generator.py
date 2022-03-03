#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


class TestTutorialTransformerGenerator(unittest.TestCase):
    def test_ppl(self):
        valid, _ = testing_utils.eval_model(
            {
                'model': 'transformer/generator',
                'model_file': 'zoo:tutorial_transformer_generator/model',
                'task': 'dailydialog',
                'skip_generation': 'true',
                'num_examples': 512,
                'batchsize': 32,
            },
            skip_test=True,
        )
        self.assertAlmostEqual(valid['ppl'], 19.59, places=2)
        self.assertAlmostEqual(valid['token_acc'], 0.4234, places=4)

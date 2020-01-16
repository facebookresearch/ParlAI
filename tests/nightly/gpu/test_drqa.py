#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


@testing_utils.skipUnlessGPU
class TestDrQAModel(unittest.TestCase):
    """
    Checks that DrQA Model can be downloaded and achieve appropriate results.
    """

    def test_pretrained(self):
        _, test = testing_utils.eval_model(
            dict(task='squad:index', model_file='zoo:drqa/squad/model')
        )
        self.assertGreaterEqual(test['accuracy'], 0.68)
        self.assertGreaterEqual(test['f1'], 0.78)


if __name__ == '__main__':
    unittest.main()

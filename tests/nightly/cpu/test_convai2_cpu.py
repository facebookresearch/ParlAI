#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


"""
This module ensures that the baseline models for convai2 produce the
correct ~~validation~~ results.

You can see the full validation set leaderboard here:
    https://github.com/DeepPavlov/convai/blob/master/leaderboards.md\
"""


@testing_utils.skipIfGPU
class TestConvai2KVMemnn(unittest.TestCase):
    """
    Checks that the KV Profile Memory model produces correct results.
    """

    def test_kvmemnn_hits1(self):
        import projects.convai2.baselines.kvmemnn.eval_hits as eval_hits

        report = eval_hits.main(args=[])
        self.assertEqual(report['hits@1'], 0.5510)

    def test_kvmemnn_f1(self):
        import projects.convai2.baselines.kvmemnn.eval_f1 as eval_f1

        report = eval_f1.main(args=[])
        self.assertAlmostEqual(report['f1'], 0.1173, delta=0.0002)

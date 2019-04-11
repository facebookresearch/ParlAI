#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.core.testing_utils as testing_utils


"""
This module ensures that the baseline models for convai2 produce the
correct ~~validation~~ results.

You can see the full validation set leaderboard here:
    https://github.com/DeepPavlov/convai/blob/master/leaderboards.md\
"""


@testing_utils.skipUnlessGPU
class TestConvai2Seq2Seq(unittest.TestCase):
    """
    Checks that the Convai2 seq2seq model produces correct results.
    """
    def test_seq2seq_hits1(self):
        import projects.convai2.baselines.seq2seq.eval_hits as eval_hits

        with testing_utils.capture_output() as stdout:
            report = eval_hits.main()
        self.assertEqual(report['hits@1'], .1250, str(stdout))

    def test_seq2seq_f1(self):
        import projects.convai2.baselines.seq2seq.eval_f1 as eval_f1

        with testing_utils.capture_output() as stdout:
            report = eval_f1.main()
        self.assertEqual(report['f1'], .1682, str(stdout))


class TestConvai2KVMemnn(unittest.TestCase):
    """
    Checks that the KV Profile Memory model produces correct results.
    """
    def test_kvmemnn_hits1(self):
        import projects.convai2.baselines.kvmemnn.eval_hits as eval_hits

        with testing_utils.capture_output() as stdout:
            report = eval_hits.main()
        self.assertEqual(report['hits@1'], .5510, str(stdout))

    def test_kvmemnn_f1(self):
        import projects.convai2.baselines.kvmemnn.eval_f1 as eval_f1

        with testing_utils.capture_output() as stdout:
            report = eval_f1.main()
        self.assertEqual(report['f1'], .1173, str(stdout))


@testing_utils.skipUnlessGPU
class TestConvai2LanguageModel(unittest.TestCase):
    """
    Checks that the language model produces correct results.
    """
    def test_languagemodel_f1(self):
        import projects.convai2.baselines.language_model.eval_f1 as eval_f1

        with testing_utils.capture_output() as stdout:
            report = eval_f1.main()
        self.assertEqual(report['f1'], .1531, str(stdout))


if __name__ == '__main__':
    unittest.main()

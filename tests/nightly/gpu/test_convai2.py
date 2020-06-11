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


@unittest.skip("Disabled due to LSTM CUDNN bug. (#2436)")
class TestConvai2Seq2Seq(unittest.TestCase):
    """
    Checks that the Convai2 seq2seq model produces correct results.
    """

    def test_seq2seq_hits1(self):
        import projects.convai2.baselines.seq2seq.eval_hits as eval_hits

        report = eval_hits.main(args=[])
        self.assertAlmostEqual(report['hits@1'], 0.1247, places=4)

    def test_seq2seq_f1(self):
        import projects.convai2.baselines.seq2seq.eval_f1 as eval_f1

        report = eval_f1.main(args=[])
        self.assertAlmostEqual(report['f1'], 0.1682, places=4)


@testing_utils.skipUnlessGPU
class TestConvai2LanguageModel(unittest.TestCase):
    """
    Checks that the language model produces correct results.
    """

    def test_languagemodel_f1(self):
        import projects.convai2.baselines.language_model.eval_f1 as eval_f1

        report = eval_f1.main(args=[])
        self.assertAlmostEqual(report['f1'], 0.1531, places=4)


class TestLegacyVersioning(unittest.TestCase):
    @testing_utils.skipUnlessGPU
    def test_legacy_version(self):
        # simply tries to load and run some models with versioning attached
        with self.assertRaises(RuntimeError):
            testing_utils.display_model(
                {
                    'model_file': 'models:convai2/seq2seq/convai2_self_seq2seq_model',
                    'task': 'convai2',
                    'no_cuda': True,
                }
            )

        testing_utils.display_model(
            {
                'model': 'legacy:seq2seq:0',
                'model_file': 'models:convai2/seq2seq/convai2_self_seq2seq_model',
                'task': 'convai2',
                'no_cuda': True,
            }
        )


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


@testing_utils.skipUnlessGPU
class TestBertModel(unittest.TestCase):
    """
    Test of Bert biencoder and crossencoder.

    Checks that Both Biencoder and CrossEncoder of Bert can be trained for about 100
    samples on convai2
    """

    @testing_utils.retry(ntries=3, log_retry=True)
    def test_biencoder(self):
        valid, test = testing_utils.train_model(
            dict(
                task='convai2',
                model='bert_ranker/bi_encoder_ranker',
                num_epochs=0.1,
                batchsize=8,
                learningrate=3e-4,
                text_truncate=32,
                validation_max_exs=20,
                short_final_eval=True,
            )
        )
        # can't conclude much from the biencoder after that little iterations.
        # this test will just make sure it hasn't crashed and the accuracy isn't
        # too high
        self.assertLessEqual(test['accuracy'], 0.5)

    @testing_utils.retry(ntries=3, log_retry=True)
    def test_crossencoder(self):
        valid, test = testing_utils.train_model(
            dict(
                task='convai2',
                model='bert_ranker/cross_encoder_ranker',
                num_epochs=0.002,
                batchsize=1,
                candidates="inline",
                type_optimization="all_encoder_layers",
                warmup_updates=100,
                text_truncate=32,
                label_truncate=32,
                validation_max_exs=20,
                short_final_eval=True,
            )
        )
        # The cross encoder reaches an interesting state MUCH faster
        # accuracy should be present and somewhere between 0.2 and 0.8
        # (large interval so that it doesn't flake.)
        self.assertGreaterEqual(test['accuracy'], 0.03)
        self.assertLessEqual(test['accuracy'], 0.8)


if __name__ == '__main__':
    unittest.main()

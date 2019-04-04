#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.core.testing_utils as testing_utils


@testing_utils.skipUnlessGPU
class TestBertModel(unittest.TestCase):
    """
    Test of Bert biencoder and crossencoder.

    Checks that Both Biencoder and CrossEncoder of Bert can be trained
    for about 100 samples on convai2
    """

    def test_biencoder(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='convai2',
            model='bert_ranker/bi_encoder_ranker',
            num_epochs=0.1,
            batchsize=8,
            text_truncate=32,
            validation_max_exs=20,
            short_final_eval=True,
        ))
        # can't conclude much from the biencoder after that little iterations.
        # accuracy should be present and somewhere between 0.01 and 0.2
        # basically it's still a random classifier
        self.assertGreaterEqual(
            test['accuracy'], 0.01,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout)
        )
        self.assertLessEqual(
            test['accuracy'], 0.2,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout)
        )

    def test_crossencoder(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='convai2',
            model='bert_ranker/cross_encoder_ranker',
            num_epochs=0.001,
            batchsize=1,
            candidates="inline",
            type_optimization="all_encoder_layers",
            warmup_updates=100,
            text_truncate=32,
            label_truncate=32,
            validation_max_exs=20,
            short_final_eval=True,
        ))
        # The cross encoder reaches an interesting state MUCH faster
        # accuracy should be present and somewhere between 0.2 and 0.8
        # (large interval so that it doesn't flake.)
        self.assertGreaterEqual(
            test['accuracy'], 0.15,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout)
        )
        self.assertLessEqual(
            test['accuracy'], 0.8,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout)
        )


if __name__ == '__main__':
    unittest.main()

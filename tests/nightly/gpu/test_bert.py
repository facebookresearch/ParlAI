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

    def test_biencoder(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:overfit',
                model='bert_ranker/bi_encoder_ranker',
                max_train_steps=500,
                batchsize=2,
                candidates="inline",
                gradient_clip=1.0,
                learningrate=1e-3,
                text_truncate=8,
            )
        )
        self.assertGreaterEqual(test['accuracy'], 0.5)

    def test_crossencoder(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:overfit',
                model='bert_ranker/cross_encoder_ranker',
                max_train_steps=500,
                batchsize=2,
                learningrate=1e-3,
                gradient_clip=1.0,
                text_truncate=8,
                label_truncate=8,
            )
        )
        self.assertGreaterEqual(test['accuracy'], 0.8)

    def test_bertclassifier(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:classifier',
                model='bert_classifier/bert_classifier',
                num_epochs=2,
                batchsize=2,
                learningrate=1e-2,
                gradient_clip=1.0,
                classes=["zero", "one"],
            )
        )
        self.assertGreaterEqual(test['accuracy'], 0.9)

    def test_bertclassifier_with_relu(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:classifier',
                model='bert_classifier/bert_classifier',
                num_epochs=2,
                batchsize=2,
                learningrate=1e-2,
                gradient_clip=1.0,
                classes=["zero", "one"],
                classifier_layers=["linear,64", "linear,2", "relu"],
            )
        )
        self.assertGreaterEqual(test['accuracy'], 0.9)


if __name__ == '__main__':
    unittest.main()

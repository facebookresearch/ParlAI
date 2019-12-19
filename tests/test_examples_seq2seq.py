#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils

BATCH_SIZE = 16
NUM_EPOCHS = 10


class TestExampleSeq2Seq(unittest.TestCase):
    """
    Checks that the example seq2seq generator model gives the expected ppl when trained
    on ConvAI2.
    """

    @testing_utils.retry(ntries=3)
    def test_generation(self):
        stdout, valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:nocandidate',
                model='examples/seq2seq',
                batchsize=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                truncate=128,
                no_cuda=True,
                embeddingsize=16,
                hiddensize=16,
            )
        )

        self.assertTrue(
            valid['token_acc'] > 0.95,
            "valid bleu = {}\nLOG:\n{}".format(valid['bleu-4'], stdout),
        )
        self.assertTrue(
            test['token_acc'] > 0.95,
            "test bleu = {}\nLOG:\n{}".format(test['bleu-4'], stdout),
        )


if __name__ == '__main__':
    unittest.main()

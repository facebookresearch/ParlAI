#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils

BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1


class TestExampleSeq2Seq(unittest.TestCase):
    """
    Checks that the example seq2seq generator model gives the expected ppl.
    """

    @testing_utils.retry(ntries=3)
    def test_generation(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:nocandidate',
                model='examples/seq2seq',
                learningrate=LR,
                batchsize=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                numthreads=1,
                hidden_size=16,
                gradient_clip=1.0,
                skip_generation=True,
            )
        )

        self.assertTrue(
            valid['token_acc'] > 0.8, "valid token_acc = {}".format(valid['token_acc']),
        )
        self.assertTrue(
            test['token_acc'] > 0.8, "test token_acc = {}".format(test['token_acc']),
        )

    @testing_utils.retry(ntries=3)
    def test_repeater(self):
        """
        Test a simple TRA based bag-of-words model.
        """
        stdout, valid, test = testing_utils.train_model(
            dict(
                task='integration_tests',
                model='examples/tra',
                num_epochs=NUM_EPOCHS,
                batchsize=BATCH_SIZE,
            )
        )

        self.assertTrue(
            valid['accuracy'] > 0.8,
            "valid accuracy = {}\nLOG:\n{}".format(valid['accuracy'], stdout),
        )
        self.assertTrue(
            test['accuracy'] > 0.8,
            "test accuracy = {}\nLOG:\n{}".format(test['accuracy'], stdout),
        )
        self.assertEqual(
            test['exs'],
            100,
            'test examples = {}\nLOG:\n{}'.format(valid['exs'], stdout),
        )


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils

BATCH_SIZE = 32
NUM_EPOCHS = 2


class TestExampleSeq2Seq(unittest.TestCase):
    """
    Checks that the example seq2seq generator model gives the expected ppl when trained
    on ConvAI2.
    """

    @testing_utils.retry(ntries=3)
    def test_generation(self):
        stdout, valid, test = testing_utils.train_model(
            dict(
                model='examples/seq2seq',
                task='convai2',
                batchsize=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                truncate=128,
            )
        )

        self.assertTrue(
            valid['ppl'] < 150, "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertTrue(
            test['ppl'] < 150, "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )


if __name__ == '__main__':
    unittest.main()

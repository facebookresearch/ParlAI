#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.core.testing_utils as testing_utils

SKIP_TESTS = False
try:
    import fairseq  # noqa: F401
except ImportError:
    SKIP_TESTS = True


BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 1e-2


class TestFairseq(unittest.TestCase):
    """Checks that fairseq can learn some very basic tasks."""

    @testing_utils.skipUnlessGPU
    @unittest.skipIf(SKIP_TESTS, "Fairseq not installed")
    def test_labelcands(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:candidate',
            model='fairseq',
            arch='lstm_wiseman_iwslt_de_en',
            lr=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            rank_candidates=True,
            skip_generation=True,
        ))

        self.assertTrue(
            valid['hits@1'] > 0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertTrue(
            test['hits@1'] > 0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )

    @testing_utils.skipUnlessGPU
    @unittest.skipIf(SKIP_TESTS, "Fairseq not installed")
    def test_generation(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:nocandidate',
            model='fairseq',
            arch='lstm_wiseman_iwslt_de_en',
            lr=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            rank_candidates=False,
            skip_generation=False,
        ))

        self.assertTrue(
            valid['ppl'] < 1.2,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertTrue(
            test['ppl'] < 1.2,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )


if __name__ == '__main__':
    unittest.main()

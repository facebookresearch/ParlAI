#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils

BATCH_SIZE = 1
NUM_EPOCHS = 3
LR = 1


class TestMemnn(unittest.TestCase):
    """
    Checks that seq2seq can learn some very basic tasks.
    """

    def test_labelcands_nomemnn(self):
        """
        This test uses a single-turn task, so doesn't test memories.
        """

        stdout, valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:candidate',
                model='memnn',
                lr=LR,
                batchsize=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                numthreads=1,
                no_cuda=True,
                embedding_size=32,
                gradient_clip=1.0,
                hops=1,
                position_encoding=True,
                use_time_features=False,
                memsize=0,
                rank_candidates=True,
            )
        )

        self.assertTrue(
            valid['hits@1'] > 0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )
        self.assertTrue(
            test['hits@1'] > 0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout),
        )

    @testing_utils.skipIfGPU
    def test_labelcands_multi(self):
        """
        This test uses a multi-turn task and multithreading.
        """
        stdout, valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='memnn',
                lr=LR,
                batchsize=BATCH_SIZE,
                num_epochs=NUM_EPOCHS * 3,
                numthreads=4,
                no_cuda=True,
                embedding_size=32,
                gradient_clip=1.0,
                hops=2,
                position_encoding=False,
                use_time_features=True,
                memsize=5,
                rank_candidates=True,
            )
        )

        self.assertTrue(
            valid['hits@1'] > 0.95,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout),
        )
        self.assertTrue(
            test['hits@1'] > 0.95,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout),
        )

    def test_backcomp(self):
        """
        Tests that the memnn model files continue to works over time.
        """
        testing_utils.download_unittest_models()

        stdout, valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests',
                model='memnn',
                model_file='zoo:unittest/memnn/model',
                dict_file='zoo:unittest/memnn/model.dict',
                batch_size=16,
            )
        )

        self.assertGreaterEqual(
            valid['accuracy'],
            0.88,
            'valid accuracy = {}\nLOG:\n{}'.format(valid['accuracy'], stdout),
        )
        self.assertGreaterEqual(
            valid['f1'], 0.999, 'valid f1 = {}\nLOG:\n{}'.format(valid['f1'], stdout)
        )
        self.assertGreaterEqual(
            test['accuracy'],
            0.84,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout),
        )
        self.assertGreaterEqual(
            test['f1'], 0.999, 'test f1 = {}\nLOG:\n{}'.format(test['f1'], stdout)
        )


if __name__ == '__main__':
    unittest.main()

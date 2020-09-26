#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils


class TestMemnn(unittest.TestCase):
    """
    Checks that seq2seq can learn some very basic tasks.
    """

    @testing_utils.retry()
    def test_labelcands_nomemnn(self):
        """
        This test uses a single-turn task, so doesn't test memories.
        """

        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:overfit',
                model='memnn',
                optimizer='sgd',
                lr=1,
                momentum=0.9,
                batchsize=4,
                num_epochs=100,
                validation_every_n_epochs=1,
                embedding_size=32,
                gradient_clip=1.0,
                hops=1,
                position_encoding=True,
                time_features=False,
                memsize=0,
                rank_candidates=True,
            )
        )

        self.assertGreater(valid['hits@1'], 0.95)
        self.assertGreater(test['hits@1'], 0.95)

    def test_backcomp(self):
        """
        Tests that the memnn model files continue to works over time.
        """
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests',
                model='memnn',
                model_file='zoo:unittest/memnn/model',
                dict_file='zoo:unittest/memnn/model.dict',
                batchsize=16,
            )
        )

        self.assertGreaterEqual(valid['accuracy'], 0.88)
        self.assertGreaterEqual(valid['f1'], 0.999)
        self.assertGreaterEqual(test['accuracy'], 0.84)
        self.assertGreaterEqual(test['f1'], 0.999)


if __name__ == '__main__':
    unittest.main()

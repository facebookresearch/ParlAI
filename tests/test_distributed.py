#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch.distributed as dist
import parlai.utils.testing as testing_utils


@testing_utils.skipUnlessGPU
class TestDistributed(unittest.TestCase):

    def test_generator_distributed(self):
        valid, test = testing_utils.distributed_train_model(
            dict(
                task='integration_tests:nocandidate',
                model='transformer/generator',
                optimizer='adamax',
                learningrate=7e-3,
                batchsize=32,
                validation_every_n_epochs=5,
                num_epochs=20,
                n_layers=1,
                n_heads=1,
                ffn_size=32,
                embedding_size=32,
                beam_size=1,
            )
        )

        self.assertLessEqual(valid['ppl'], 1.20)
        self.assertGreaterEqual(valid['bleu-4'], 0.95)
        self.assertLessEqual(test['ppl'], 1.20)
        self.assertGreaterEqual(test['bleu-4'], 0.95)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import parlai.utils.testing as testing_utils


@testing_utils.skipUnlessGPU
class TestMemEfficientFP16(unittest.TestCase):
    """
    Test memory efficient FP16 implementation.
    """

    def test_adam(self):
        valid, _ = testing_utils.train_model(
            dict(
                task='integration_tests:candidate',
                model='transformer/ranker',
                optimizer='adam',
                fp16=True,
                fp16_impl='mem_efficient',
                learningrate=7e-3,
                batchsize=32,
                num_epochs=1,
                n_layers=1,
                n_heads=1,
                ffn_size=32,
                embedding_size=32,
                warmup_updates=1,
                lr_scheduler='invsqrt',
            )
        )
        self.assertGreaterEqual(valid['hits@1'], 0.4)

    def test_unsupported(self):
        with self.assertRaises(RuntimeError):
            # SGD unsupported currently
            testing_utils.train_model(
                dict(
                    task='integration_tests:candidate',
                    model='transformer/ranker',
                    optimizer='sgd',
                    fp16=True,
                    fp16_impl='mem_efficient',
                )
            )

    def test_resuming(self):
        """
        Test resuming without FP16.
        """
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model')

            valid1, test1 = testing_utils.train_model(
                dict(
                    model_file=model_file,
                    task='integration_tests:candidate',
                    model='transformer/ranker',
                    optimizer='adam',
                    fp16=True,
                    fp16_impl='mem_efficient',
                    learningrate=7e-3,
                    batchsize=32,
                    num_epochs=1,
                    n_layers=1,
                    n_heads=1,
                    ffn_size=32,
                    embedding_size=32,
                    warmup_updates=1,
                    lr_scheduler='invsqrt',
                )
            )

            valid2, test2 = testing_utils.train_model(
                dict(
                    model_file=model_file,
                    task='integration_tests:candidate',
                    model='transformer/ranker',
                    num_epochs=1,
                    fp16=False,
                )
            )

            # make sure the number of updates is being tracked correctly
            self.assertGreater(
                valid2['total_train_updates'],
                valid1['total_train_updates'],
                'Number of updates is not increasing',
            )

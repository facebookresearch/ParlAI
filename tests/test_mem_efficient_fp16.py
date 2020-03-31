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

    @testing_utils.retry(ntries=3)
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
                num_epochs=0.25,
                n_layers=1,
                n_heads=1,
                ffn_size=32,
                embedding_size=32,
                warmup_updates=1,
                lr_scheduler='invsqrt',
            )
        )
        self.assertGreaterEqual(valid['hits@1'], 0.1)

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

    # we don't currently install apex in CircleCI for a variety of reasons
    @testing_utils.skipIfCircleCI
    def test_resuming_apex2memeff(self):
        """
        Test switching from memory efficient fp16 to apex fp16.
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
                    fp16_impl='apex',
                    learningrate=7e-3,
                    batchsize=32,
                    num_epochs=0.25,
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
                    fp16_impl='mem_efficient',
                    num_epochs=0.5,
                    fp16=True,
                )
            )

            # make sure the number of updates is being tracked correctly
            self.assertGreater(
                valid2['total_train_updates'],
                valid1['total_train_updates'],
                'Number of updates is not increasing',
            )

    # we don't currently install apex in CircleCI for a variety of reasons
    @testing_utils.skipIfCircleCI
    def test_resuming_memeff2apex(self):
        """
        Test switching from memory efficient fp16 to apex fp16.
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
                    num_epochs=0.25,
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
                    fp16_impl='apex',
                    num_epochs=0.5,
                    fp16=True,
                )
            )

            # make sure the number of updates is being tracked correctly
            self.assertGreater(
                valid2['total_train_updates'],
                valid1['total_train_updates'],
                'Number of updates is not increasing',
            )

    def test_resuming_adam(self):
        """
        Test resuming a memory efficient fp16 model from disk.
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
                    num_epochs=0.25,
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
                    num_epochs=0.5,
                    fp16=True,
                )
            )

            # make sure the number of updates is being tracked correctly
            self.assertGreater(
                valid2['total_train_updates'],
                valid1['total_train_updates'],
                'Number of updates is not increasing',
            )

    def test_resuming_fp32(self):
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
                    num_epochs=0.1,
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
                    num_epochs=0.25,
                    fp16=False,
                )
            )

            # make sure the number of updates is being tracked correctly
            self.assertGreater(
                valid2['total_train_updates'],
                valid1['total_train_updates'],
                'Number of updates is not increasing',
            )

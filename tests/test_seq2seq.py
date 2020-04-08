#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils

BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1


class TestSeq2Seq(unittest.TestCase):
    """
    Checks that seq2seq can learn some very basic tasks.
    """

    @testing_utils.retry(ntries=3)
    def test_ranking(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:candidate',
                model='seq2seq',
                learningrate=LR,
                batchsize=BATCH_SIZE,
                num_epochs=3,
                numthreads=1,
                embeddingsize=16,
                hiddensize=16,
                rnn_class='gru',
                attention='general',
                gradient_clip=1.0,
                dropout=0.0,
                lookuptable='all',
                skip_generation=True,
                rank_candidates=True,
            )
        )
        assert valid['hits@1'] >= 0.95

    def test_generation(self):
        """
        This test uses a single-turn sequence repitition task.
        """
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_nocandidate',
                model='seq2seq',
                model_file='zoo:unittest/seq2seq/model',
                dict_file='zoo:unittest/seq2seq/model.dict',
                skip_generation=False,
                inference='greedy',
                batchsize=8,
                num_examples=32,
            )
        )

        self.assertLess(valid['ppl'], 1.2)
        self.assertLess(test['ppl'], 1.2)

    def test_beamsearch(self):
        """
        Ensures beam search can generate the correct response.
        """
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_nocandidate',
                model='seq2seq',
                model_file='zoo:unittest/seq2seq/model',
                dict_file='zoo:unittest/seq2seq/model.dict',
                skip_generation=False,
                inference='beam',
                beam_size=5,
            )
        )
        self.assertGreater(valid['accuracy'], 0.95)
        self.assertGreater(test['accuracy'], 0.95)

    def test_badinput(self):
        """
        Ensures model doesn't crash on malformed inputs.
        """
        testing_utils.train_model(
            dict(
                task='integration_tests:bad_example',
                model='seq2seq',
                learningrate=LR,
                batchsize=10,
                datatype='train:ordered:stream',
                num_epochs=1,
                numthreads=1,
                embeddingsize=16,
                hiddensize=16,
                inference='greedy',
            )
        )


class TestHogwildSeq2seq(unittest.TestCase):
    @testing_utils.skipIfGPU
    def test_generation_multi(self):
        """
        This test uses a multi-turn task and multithreading.
        """
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:multiturn_nocandidate',
                model='seq2seq',
                learningrate=LR,
                batchsize=BATCH_SIZE,
                num_epochs=NUM_EPOCHS * 2,
                numthreads=2,
                no_cuda=True,
                embeddingsize=16,
                hiddensize=16,
                rnn_class='gru',
                attention='general',
                gradient_clip=1.0,
                dropout=0.0,
                lookuptable='all',
            )
        )

        self.assertLess(valid['ppl'], 1.2)
        self.assertLess(test['ppl'], 1.2)


class TestBackwardsCompatibility(unittest.TestCase):
    """
    Tests that a binary file continues to work over time.
    """

    def test_backwards_compatibility(self):
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='seq2seq',
                model_file='zoo:unittest/seq2seq/model',
                dict_file='zoo:unittest/seq2seq/model.dict',
            )
        )

        self.assertLessEqual(valid['ppl'], 1.01)
        self.assertGreaterEqual(valid['accuracy'], 0.999)
        self.assertGreaterEqual(valid['f1'], 0.999)
        self.assertLessEqual(test['ppl'], 1.01)
        self.assertGreaterEqual(test['accuracy'], 0.999)
        self.assertGreaterEqual(test['f1'], 0.999)


if __name__ == '__main__':
    unittest.main()

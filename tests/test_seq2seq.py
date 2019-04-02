#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.core.testing_utils as testing_utils

BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1


class TestSeq2Seq(unittest.TestCase):
    """Checks that seq2seq can learn some very basic tasks."""

    @testing_utils.retry(ntries=3)
    def test_ranking(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:candidate',
            model='seq2seq',
            learningrate=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            numthreads=1,
            no_cuda=True,
            embeddingsize=16,
            hiddensize=16,
            rnn_class='gru',
            attention='general',
            gradient_clip=1.0,
            dropout=0.0,
            lookuptable='all',
            rank_candidates=True,
        ))
        self.assertTrue(
            valid['hits@1'] >= 0.95,
            "hits@1 = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )

    @testing_utils.retry(ntries=3)
    def test_generation(self):
        """This test uses a single-turn sequence repitition task."""
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:nocandidate',
            model='seq2seq',
            learningrate=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            numthreads=1,
            no_cuda=True,
            embeddingsize=16,
            hiddensize=16,
            rnn_class='gru',
            attention='general',
            gradient_clip=1.0,
            dropout=0.0,
            lookuptable='all',
        ))

        self.assertTrue(
            valid['ppl'] < 1.2,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertTrue(
            test['ppl'] < 1.2,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )

    @testing_utils.retry(ntries=3)
    def test_beamsearch(self):
        """Ensures beam search can generate the correct response"""
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:nocandidate',
            model='seq2seq',
            learningrate=LR,
            batchsize=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            numthreads=1,
            no_cuda=True,
            embeddingsize=16,
            hiddensize=16,
            rnn_class='gru',
            attention='general',
            gradient_clip=1.0,
            dropout=0.0,
            lookuptable='all',
            beam_size=4,
        ))

        self.assertTrue(
            valid['bleu'] > 0.95,
            "valid bleu = {}\nLOG:\n{}".format(valid['bleu'], stdout)
        )
        self.assertTrue(
            test['bleu'] > 0.95,
            "test bleu = {}\nLOG:\n{}".format(test['bleu'], stdout)
        )
        self.assertTrue(
            valid['ppl'] < 1.2,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertTrue(
            test['ppl'] < 1.2,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )

    def test_badinput(self):
        """Ensures model doesn't crash on malformed inputs."""
        stdout, _, _ = testing_utils.train_model(dict(
            task='integration_tests:bad_example',
            model='seq2seq',
            learningrate=LR,
            batchsize=10,
            datatype='train:ordered:stream',
            num_epochs=1,
            numthreads=1,
            no_cuda=True,
            embeddingsize=16,
            hiddensize=16,
        ))
        self.assertIn('valid:{', stdout)
        self.assertIn('test:{', stdout)


class TestHogwildSeq2seq(unittest.TestCase):
    @testing_utils.skipIfGPU
    def test_generation_multi(self):
        """This test uses a multi-turn task and multithreading."""
        stdout, valid, test = testing_utils.train_model(dict(
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
        ))

        self.assertTrue(
            valid['ppl'] < 1.2,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertTrue(
            test['ppl'] < 1.2,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )


class TestBackwardsCompatibility(unittest.TestCase):
    """
    Tests that a binary file continues to work over time.
    """
    def test_backwards_compatibility(self):
        testing_utils.download_unittest_models()

        stdout, valid, test = testing_utils.eval_model(dict(
            task='integration_tests:multipass',
            model='seq2seq',
            model_file='models:unittest/seq2seq/model',
            dict_file='models:unittest/seq2seq/model.dict',
            no_cuda=True,
        ))

        self.assertLessEqual(
            valid['ppl'], 1.01,
            'valid ppl = {}\nLOG:\n{}'.format(valid['ppl'], stdout),
        )
        self.assertGreaterEqual(
            valid['accuracy'], .999,
            'valid accuracy = {}\nLOG:\n{}'.format(valid['accuracy'], stdout),
        )
        self.assertGreaterEqual(
            valid['f1'], .999,
            'valid f1 = {}\nLOG:\n{}'.format(valid['f1'], stdout)
        )
        self.assertLessEqual(
            test['ppl'], 1.01,
            'test ppl = {}\nLOG:\n{}'.format(test['ppl'], stdout),
        )
        self.assertGreaterEqual(
            test['accuracy'], .999,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout),
        )
        self.assertGreaterEqual(
            test['f1'], .999,
            'test f1 = {}\nLOG:\n{}'.format(test['f1'], stdout)
        )


if __name__ == '__main__':
    unittest.main()

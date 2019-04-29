#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import parlai.core.testing_utils as testing_utils


class TestTransformerRanker(unittest.TestCase):
    """Checks that transformer_ranker can learn some very basic tasks."""

    @testing_utils.retry(ntries=3)
    def test_repeater(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:candidate',
            model='transformer/ranker',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=16,
            validation_every_n_epochs=5,
            validation_patience=2,
            n_layers=1,
            n_heads=4,
            ffn_size=64,
            embedding_size=32,
            candidates='batch',
            eval_candidates='inline',
            gradient_clip=0.5,
        ))

        self.assertGreaterEqual(
            valid['hits@1'],
            0.90,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertGreaterEqual(
            test['hits@1'],
            0.90,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )

    def test_resuming(self):
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model')

            stdout1, valid1, test1 = testing_utils.train_model(dict(
                model_file=model_file,
                task='integration_tests:candidate',
                model='transformer/ranker',
                optimizer='adamax',
                learningrate=7e-3,
                batchsize=32,
                num_epochs=1,
                n_layers=1,
                n_heads=1,
                ffn_size=32,
                embedding_size=32,
                warmup_updates=1,
                lr_scheduler='invsqrt',
            ))

            stdout2, valid2, test2 = testing_utils.train_model(dict(
                model_file=model_file,
                task='integration_tests:candidate',
                model='transformer/ranker',
                num_epochs=1,
            ))
            # make sure the number of updates is being tracked correctly
            self.assertGreater(
                valid2['num_updates'],
                valid1['num_updates'],
                'Number of updates is not increasing'
            )
            # make sure the learning rate is decreasing
            self.assertLess(
                valid2['lr'],
                valid1['lr'],
                'Learning rate is not decreasing'
            )

    def test_backcomp(self):
        """
        Tests that the transformer ranker model files continue to work over time.
        """
        testing_utils.download_unittest_models()

        stdout, valid, test = testing_utils.eval_model(dict(
            task='integration_tests:multipass',
            model='transformer/ranker',
            model_file='models:unittest/transformer_ranker/model',
            dict_file='models:unittest/transformer_ranker/model.dict',
            batch_size=64,
        ))

        self.assertGreaterEqual(
            valid['hits@1'], .99,
            'valid hits@1 = {}\nLOG:\n{}'.format(valid['hits@1'], stdout),
        )
        self.assertGreaterEqual(
            valid['accuracy'], .99,
            'valid accuracy = {}\nLOG:\n{}'.format(valid['accuracy'], stdout),
        )
        self.assertGreaterEqual(
            valid['f1'], .99,
            'valid f1 = {}\nLOG:\n{}'.format(valid['f1'], stdout)
        )
        self.assertGreaterEqual(
            test['hits@1'], .99,
            'test hits@1 = {}\nLOG:\n{}'.format(test['hits@1'], stdout),
        )
        self.assertGreaterEqual(
            test['accuracy'], .99,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout),
        )
        self.assertGreaterEqual(
            test['f1'], .99,
            'test f1 = {}\nLOG:\n{}'.format(test['f1'], stdout)
        )

    @testing_utils.retry(ntries=3)
    def test_xlm(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:candidate',
            model='transformer/ranker',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=16,
            validation_every_n_epochs=5,
            validation_patience=2,
            n_layers=1,
            n_heads=4,
            ffn_size=64,
            embedding_size=32,
            candidates='batch',
            eval_candidates='inline',
            gradient_clip=0.5,
            variant='xlm',
            activation='gelu',
        ))

        self.assertGreaterEqual(
            valid['hits@1'],
            0.90,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertGreaterEqual(
            test['hits@1'],
            0.90,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )

    @testing_utils.retry(ntries=3)
    def test_alt_reduction(self):
        """Test a transformer ranker reduction method other than `mean`."""
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:candidate',
            model='transformer/ranker',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=16,
            validation_every_n_epochs=5,
            validation_patience=2,
            n_layers=1,
            n_heads=4,
            ffn_size=64,
            embedding_size=32,
            candidates='batch',
            eval_candidates='inline',
            gradient_clip=0.5,
            variant='xlm',
            activation='gelu',
            reduction_type='first',  # this is really what we're trying to test for
        ))

        self.assertGreaterEqual(
            valid['hits@1'],
            0.90,
            "valid hits@1 = {}\nLOG:\n{}".format(valid['hits@1'], stdout)
        )
        self.assertGreaterEqual(
            test['hits@1'],
            0.90,
            "test hits@1 = {}\nLOG:\n{}".format(test['hits@1'], stdout)
        )


class TestTransformerGenerator(unittest.TestCase):
    """Checks that the generative transformer can learn basic tasks."""
    @testing_utils.retry(ntries=3)
    def test_greedysearch(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:nocandidate',
            model='transformer/generator',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=32,
            num_epochs=20,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            beam_size=1,
        ))

        self.assertLessEqual(
            valid['ppl'],
            1.30,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertGreaterEqual(
            valid['bleu'],
            0.90,
            "valid blue = {}\nLOG:\n{}".format(valid['bleu'], stdout)
        )
        self.assertLessEqual(
            test['ppl'],
            1.30,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )
        self.assertGreaterEqual(
            test['bleu'],
            0.90,
            "test bleu = {}\nLOG:\n{}".format(test['bleu'], stdout)
        )

    @testing_utils.retry(ntries=3)
    def test_beamsearch(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:nocandidate',
            model='transformer/generator',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=32,
            num_epochs=20,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            beam_size=5,
        ))

        self.assertLessEqual(
            valid['ppl'],
            1.20,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertGreaterEqual(
            valid['bleu'],
            0.95,
            "valid blue = {}\nLOG:\n{}".format(valid['bleu'], stdout)
        )
        self.assertLessEqual(
            test['ppl'],
            1.20,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )
        self.assertGreaterEqual(
            test['bleu'],
            0.95,
            "test bleu = {}\nLOG:\n{}".format(test['bleu'], stdout)
        )

    def test_generator_backcomp(self):
        """
        Tests that the generator model files work over time.
        """
        testing_utils.download_unittest_models()

        stdout, valid, test = testing_utils.eval_model(dict(
            task='integration_tests:multipass',
            model='transformer/generator',
            model_file='models:unittest/transformer_generator2/model',
            dict_file='models:unittest/transformer_generator2/model.dict',
            rank_candidates=True,
            batch_size=64,
        ))

        self.assertGreaterEqual(
            valid['hits@1'], 0.95,
            'valid hits@1 = {}\nLOG:\n{}'.format(valid['hits@1'], stdout),
        )
        self.assertLessEqual(
            valid['ppl'], 1.01,
            'valid ppl = {}\nLOG:\n{}'.format(valid['ppl'], stdout),
        )
        self.assertGreaterEqual(
            valid['accuracy'], .99,
            'valid accuracy = {}\nLOG:\n{}'.format(valid['accuracy'], stdout),
        )
        self.assertGreaterEqual(
            valid['f1'], .99,
            'valid f1 = {}\nLOG:\n{}'.format(valid['f1'], stdout)
        )
        self.assertGreaterEqual(
            test['hits@1'], 0.95,
            'test hits@1 = {}\nLOG:\n{}'.format(test['hits@1'], stdout),
        )
        self.assertLessEqual(
            test['ppl'], 1.01,
            'test ppl = {}\nLOG:\n{}'.format(test['ppl'], stdout),
        )
        self.assertGreaterEqual(
            test['accuracy'], .99,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout),
        )
        self.assertGreaterEqual(
            test['f1'], .99,
            'test f1 = {}\nLOG:\n{}'.format(test['f1'], stdout)
        )

    def test_badinput(self):
        """Ensures model doesn't crash on malformed inputs."""
        stdout, _, _ = testing_utils.train_model(dict(
            task='integration_tests:bad_example',
            model='transformer/generator',
            batchsize=10,
            datatype='train:ordered:stream',
            num_epochs=1,
            numthreads=1,
            no_cuda=True,
            embedding_size=16,
            hiddensize=16,
        ))
        self.assertIn('valid:{', stdout)
        self.assertIn('test:{', stdout)

    @testing_utils.retry(ntries=3)
    def test_xlm(self):
        stdout, valid, test = testing_utils.train_model(dict(
            task='integration_tests:nocandidate',
            model='transformer/generator',
            optimizer='adamax',
            learningrate=7e-3,
            batchsize=32,
            num_epochs=20,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            beam_size=1,
            variant='xlm',
            activation='gelu',
            n_segments=8,  # doesn't do anything but still good to test
        ))

        self.assertLessEqual(
            valid['ppl'],
            1.30,
            "valid ppl = {}\nLOG:\n{}".format(valid['ppl'], stdout)
        )
        self.assertGreaterEqual(
            valid['bleu'],
            0.90,
            "valid blue = {}\nLOG:\n{}".format(valid['bleu'], stdout)
        )
        self.assertLessEqual(
            test['ppl'],
            1.30,
            "test ppl = {}\nLOG:\n{}".format(test['ppl'], stdout)
        )
        self.assertGreaterEqual(
            test['bleu'],
            0.90,
            "test bleu = {}\nLOG:\n{}".format(test['bleu'], stdout)
        )


class TestLearningRateScheduler(unittest.TestCase):
    def test_resuming(self):
        BASE_ARGS = dict(
            task='integration_tests:nocandidate',
            model='transformer/generator',
            optimizer='adamax',
            learningrate=1e-3,
            batchsize=32,
            num_epochs=1,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            skip_generation=True,
            warmup_updates=1,
        )

        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model')

            stdout1, valid1, test1 = testing_utils.train_model(dict(
                model_file=model_file,
                lr_scheduler='invsqrt',
                **BASE_ARGS,
            ))
            stdout2, valid2, test2 = testing_utils.train_model(dict(
                model_file=model_file,
                lr_scheduler='invsqrt',
                **BASE_ARGS,
            ))
            # make sure the number of updates is being tracked correctly
            self.assertGreater(
                valid2['num_updates'],
                valid1['num_updates'],
                'Number of updates is not increasing'
            )
            # make sure the learning rate is decreasing
            self.assertLess(
                valid2['lr'],
                valid1['lr'],
                'Learning rate is not decreasing'
            )
            # but make sure we're not loading the scheduler if we're fine tuning
            stdout3, valid3, test3 = testing_utils.train_model(dict(
                init_model=os.path.join(tmpdir, 'model'),
                model_file=os.path.join(tmpdir, 'newmodel'),
                lr_scheduler='invsqrt',
                **BASE_ARGS,
            ))
            self.assertEqual(
                valid3['num_updates'],
                valid1['num_updates'],
                'Finetuning LR scheduler reset failed (num_updates).'
            )
            self.assertEqual(
                valid3['lr'],
                valid1['lr'],
                'Finetuning LR scheduler reset failed (lr).'
            )
            # and make sure we're not loading the scheduler if it changes
            stdout4, valid4, test4 = testing_utils.train_model(dict(
                init_model=os.path.join(tmpdir, 'model'),
                model_file=os.path.join(tmpdir, 'newmodel2'),
                lr_scheduler='reduceonplateau',
                **BASE_ARGS
            ))
            self.assertEqual(
                valid4['num_updates'],
                valid1['num_updates'],
                'LR scheduler change reset failed (num_updates).\n' + stdout4
            )
            self.assertEqual(
                valid4['lr'],
                1e-3,
                'LR is not correct in final resume.\n' + stdout4
            )


if __name__ == '__main__':
    unittest.main()

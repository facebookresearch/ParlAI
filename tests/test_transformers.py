#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test many variants of transformers.
"""

import os
import unittest
import parlai.utils.testing as testing_utils
from parlai.core.agents import create_agent
from parlai.core.opt import Opt


class TestTransformerRanker(unittest.TestCase):
    """
    Checks that transformer_ranker can learn some very basic tasks.
    """

    @testing_utils.retry(ntries=3)
    def test_repeater(self):
        """
        Test a simple repeat-after-me model.
        """
        valid, test = testing_utils.train_model(
            dict(
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
            )
        )

        self.assertGreaterEqual(valid['hits@1'], 0.90)
        self.assertGreaterEqual(test['hits@1'], 0.90)

    def test_resuming(self):
        """
        Test saving and resuming training.
        """
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model')

            valid1, test1 = testing_utils.train_model(
                dict(
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
                )
            )

            valid2, test2 = testing_utils.train_model(
                dict(
                    model_file=model_file,
                    task='integration_tests:candidate',
                    model='transformer/ranker',
                    num_epochs=1,
                )
            )
            # make sure the number of updates is being tracked correctly
            self.assertGreater(
                valid2['total_train_updates'],
                valid1['total_train_updates'],
                'Number of updates is not increasing',
            )
            # make sure the learning rate is decreasing
            self.assertLess(
                valid2['lr'], valid1['lr'], 'Learning rate is not decreasing'
            )

    def test_resuming_reduce_on_plateau(self):
        """
        Reduce on Plateau can be tricky when combined with warmup.

        See: https://github.com/facebookresearch/ParlAI/pull/1812
        """
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model')
            valid1, test1 = testing_utils.train_model(
                dict(
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
                    lr_scheduler='reduceonplateau',
                )
            )

            valid2, test2 = testing_utils.train_model(
                dict(
                    model_file=model_file,
                    task='integration_tests:candidate',
                    model='transformer/ranker',
                    num_epochs=1,
                    lr_scheduler='reduceonplateau',
                )
            )
            # make sure the learning rate is decreasing
            self.assertGreater(
                valid2['lr'], 1e-5, 'Learning rate should not be that low when resuming'
            )

    def test_backcomp(self):
        """
        Tests that the transformer ranker model files continue to work over time.
        """
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='transformer/ranker',
                model_file='zoo:unittest/transformer_ranker/model',
                dict_file='zoo:unittest/transformer_ranker/model.dict',
                batch_size=64,
            )
        )

        self.assertGreaterEqual(valid['hits@1'], 0.99)
        self.assertGreaterEqual(valid['accuracy'], 0.99)
        self.assertGreaterEqual(valid['f1'], 0.99)
        self.assertGreaterEqual(test['hits@1'], 0.99)
        self.assertGreaterEqual(test['accuracy'], 0.99)
        self.assertGreaterEqual(test['f1'], 0.99)

    @testing_utils.retry(ntries=3)
    def test_xlm(self):
        """
        Test --variant xlm.
        """
        valid, test = testing_utils.train_model(
            dict(
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
            )
        )

        self.assertGreaterEqual(valid['hits@1'], 0.90)
        self.assertGreaterEqual(test['hits@1'], 0.90)

    @testing_utils.retry(ntries=3)
    def test_alt_reduction(self):
        """
        Test a transformer ranker reduction method other than `mean`.
        """
        valid, test = testing_utils.train_model(
            dict(
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
            )
        )

        self.assertGreaterEqual(valid['hits@1'], 0.90)
        self.assertGreaterEqual(test['hits@1'], 0.90)


class TestTransformerGenerator(unittest.TestCase):
    """
    Checks that the generative transformer can learn basic tasks.
    """

    @testing_utils.retry(ntries=3)
    def test_greedysearch(self):
        """
        Test greedy search.
        """
        valid, test = testing_utils.train_model(
            dict(
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
                inference='greedy',
                beam_size=1,
            )
        )

        self.assertLessEqual(valid['ppl'], 1.30)
        self.assertGreaterEqual(valid['bleu-4'], 0.90)
        self.assertLessEqual(test['ppl'], 1.30)
        self.assertGreaterEqual(test['bleu-4'], 0.90)

    @testing_utils.retry(ntries=3)
    def test_beamsearch(self):
        """
        Test beamsearch.
        """
        valid, test = testing_utils.train_model(
            dict(
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
                inference='beam',
                beam_size=5,
            )
        )

        self.assertLessEqual(valid['ppl'], 1.20)
        self.assertGreaterEqual(valid['bleu-4'], 0.95)
        self.assertLessEqual(test['ppl'], 1.20)
        self.assertGreaterEqual(test['bleu-4'], 0.95)

    @testing_utils.retry(ntries=3)
    def test_beamsearch_blocking(self):
        """
        Test beamsearch blocking.
        """
        with testing_utils.tempdir() as tmpdir:
            mf = os.path.join(tmpdir, 'model')
            df = os.path.join(tmpdir, 'model.dict')
            valid, test = testing_utils.train_model(
                dict(
                    task='integration_tests:repeat_words',
                    model='transformer/generator',
                    model_file=mf,
                    dict_file=df,
                    optimizer='adamax',
                    learningrate=7e-3,
                    batchsize=32,
                    num_epochs=20,
                    n_layers=1,
                    n_heads=1,
                    ffn_size=32,
                    embedding_size=32,
                    inference='beam',
                    beam_size=2,
                )
            )
            valid_beam_block, test_beam_block = testing_utils.eval_model(
                dict(
                    task='integration_tests:repeat_words',
                    model_file=mf,
                    dict_file=df,
                    batch_size=1,
                    inference='beam',
                    beam_size=5,
                    beam_block_ngram=1,
                    skip_generation=False,
                )
            )
            valid_beam_block2, test_beam_block2 = testing_utils.eval_model(
                dict(
                    task='integration_tests:repeat_words',
                    model_file=mf,
                    dict_file=df,
                    batch_size=1,
                    inference='beam',
                    beam_size=5,
                    beam_block_ngram=2,
                    skip_generation=False,
                )
            )
        self.assertLessEqual(valid['ppl'], 1.30)
        self.assertGreaterEqual(valid['f1'], 0.80)
        self.assertGreaterEqual(valid['bleu-4'], 0.5)
        self.assertLessEqual(test['ppl'], 1.30)
        self.assertGreaterEqual(test['f1'], 0.80)
        self.assertGreaterEqual(test['bleu-4'], 0.5)

        # Beam Block 1
        self.assertLessEqual(valid_beam_block['f1'], 0.4)
        self.assertLessEqual(valid_beam_block['bleu-4'], 1e-9)
        self.assertLessEqual(test_beam_block['f1'], 0.4)
        self.assertLessEqual(test_beam_block['bleu-4'], 1e-9)

        # Beam Block 2
        self.assertLessEqual(valid_beam_block2['f1'], 0.6)
        self.assertLessEqual(valid_beam_block2['bleu-4'], 1e-6)
        self.assertLessEqual(test_beam_block2['f1'], 0.6)
        self.assertLessEqual(test_beam_block2['bleu-4'], 1e-6)

    @testing_utils.retry(ntries=3)
    def test_beamsearch_contextblocking(self):
        """
        Test beamsearch context blocking.

        General strategy: train a parrot model, then block it from doing the parroting
        well. Measure how much context blocking affects performance.
        """

        with testing_utils.tempdir() as tmpdir:
            mf = os.path.join(tmpdir, 'model')
            df = os.path.join(tmpdir, 'model.dict')
            # we'll reuse these
            args = dict(
                task='integration_tests', model_file=mf, dict_file=df, metrics='all'
            )
            noblock_valid, _ = testing_utils.train_model(
                dict(
                    model='transformer/generator',
                    optimizer='adamax',
                    learningrate=7e-3,
                    batchsize=32,
                    num_epochs=20,
                    n_layers=1,
                    n_heads=1,
                    ffn_size=32,
                    embedding_size=32,
                    inference='beam',
                    beam_size=5,
                    **args,
                )
            )
            self.assertGreaterEqual(noblock_valid['f1'], 0.95)

            # first confirm all is good without blocking
            valid, test = testing_utils.eval_model(
                dict(beam_context_block_ngram=-1, **args)
            )
            self.assertGreaterEqual(valid['f1'], 0.95)
            self.assertGreaterEqual(valid['bleu-4'], 0.95)

            # there's a special case for block == 1
            valid, test = testing_utils.eval_model(
                dict(beam_context_block_ngram=1, **args)
            )
            # bleu and f1 should be totally wrecked.
            self.assertLess(valid['f1'], 0.01)
            self.assertLess(valid['bleu-4'], 0.01)

            # a couple general cases
            valid, test = testing_utils.eval_model(
                dict(beam_context_block_ngram=2, **args)
            )
            # should take a big hit here
            self.assertLessEqual(valid['f1'], noblock_valid['f1'])
            # bleu-1 should be relatively okay
            self.assertLessEqual(valid['bleu-1'], noblock_valid['bleu-1'])
            self.assertGreaterEqual(valid['bleu-1'], 0.45)
            # and bleu-2 should be 0 at this point
            self.assertLessEqual(valid['bleu-2'], 0.01)

            # larger blocking, we can do better now
            valid, test = testing_utils.eval_model(
                dict(beam_context_block_ngram=3, **args)
            )
            # not as hard a hit from the larger hit
            self.assertLessEqual(valid['f1'], 0.95)
            # bleu-1 and bleu-2 should be relatively okay
            self.assertGreaterEqual(valid['bleu-1'], 0.60)
            self.assertGreaterEqual(valid['bleu-2'], 0.25)
            # bleu-3 should be totally screwed
            self.assertLessEqual(valid['bleu-3'], 0.01)

    def test_nucleus(self):
        """
        Test nucleus generation.
        """
        # Nucleus is inherently stochastic, just ensure no crash.
        testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='transformer/generator',
                model_file='zoo:unittest/transformer_generator2/model',
                batchsize=32,
                inference='nucleus',
                topp=0.3,
            )
        )

    def test_topk(self):
        """
        Test topk generation.
        """
        # Topk is inherently stochastic, just ensure no crash.
        testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='transformer/generator',
                model_file='zoo:unittest/transformer_generator2/model',
                batchsize=32,
                inference='topk',
                topk=5,
            )
        )

    def test_generator_backcomp(self):
        """
        Tests that the generator model files work over time.
        """
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='transformer/generator',
                model_file='zoo:unittest/transformer_generator2/model',
                dict_file='zoo:unittest/transformer_generator2/model.dict',
                rank_candidates=True,
                batch_size=64,
            )
        )

        self.assertGreaterEqual(valid['hits@1'], 0.95)
        self.assertLessEqual(valid['ppl'], 1.01)
        self.assertGreaterEqual(valid['accuracy'], 0.99)
        self.assertGreaterEqual(valid['f1'], 0.99)
        self.assertGreaterEqual(test['hits@1'], 0.95)
        self.assertLessEqual(test['ppl'], 1.01)
        self.assertGreaterEqual(test['accuracy'], 0.99)
        self.assertGreaterEqual(test['f1'], 0.99)

    def test_badinput(self):
        """
        Ensures model doesn't crash on malformed inputs.
        """
        testing_utils.train_model(
            dict(
                task='integration_tests:bad_example',
                model='transformer/generator',
                batchsize=10,
                datatype='train:ordered:stream',
                num_epochs=1,
                numthreads=1,
                no_cuda=True,
                embedding_size=16,
                hiddensize=16,
            )
        )

    @testing_utils.retry(ntries=3)
    def test_xlm(self):
        """
        Test --variant xlm.
        """
        valid, test = testing_utils.train_model(
            dict(
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
                inference='greedy',
                beam_size=1,
                variant='xlm',
                activation='gelu',
                n_segments=8,  # doesn't do anything but still good to test
                adam_eps=1e-6,  # just to test another flag simultaneously
            )
        )

        self.assertLessEqual(valid['ppl'], 1.30)
        self.assertGreaterEqual(valid['bleu-4'], 0.90)
        self.assertLessEqual(test['ppl'], 1.30)
        self.assertGreaterEqual(test['bleu-4'], 0.90)

    def test_asymmetry(self):
        opt = Opt({'model': 'transformer/generator', 'n_layers': 1})
        agent = create_agent(opt)
        self.assertEqual(agent.model.encoder.n_layers, 1)
        self.assertEqual(agent.model.decoder.n_layers, 1)

        opt = Opt(
            {'model': 'transformer/generator', 'n_layers': 1, 'n_encoder_layers': 2}
        )
        agent = create_agent(opt)
        self.assertEqual(agent.model.encoder.n_layers, 2)
        self.assertEqual(agent.model.decoder.n_layers, 1)

        opt = Opt(
            {
                'model': 'transformer/generator',
                'n_layers': 1,
                'n_encoder_layers': 2,
                'n_decoder_layers': 4,
            }
        )
        agent = create_agent(opt)
        self.assertEqual(agent.model.encoder.n_layers, 2)
        self.assertEqual(agent.model.decoder.n_layers, 4)

        opt = Opt(
            {'model': 'transformer/generator', 'n_layers': 1, 'n_decoder_layers': 4}
        )
        agent = create_agent(opt)
        self.assertEqual(agent.model.encoder.n_layers, 1)
        self.assertEqual(agent.model.decoder.n_layers, 4)

        opt = Opt({'model': 'transformer/generator'})
        agent = create_agent(opt)
        self.assertEqual(agent.model.encoder.n_layers, 2)
        self.assertEqual(agent.model.decoder.n_layers, 2)


class TestLearningRateScheduler(unittest.TestCase):
    """
    Test learning rate scheduler for both generative and ranking transformers.
    """

    def _test_learning_rate_resuming(self, args):
        """
        Test learning rate resumes correctly.
        """
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model')
            valid1, test1 = testing_utils.train_model(
                dict(model_file=model_file, lr_scheduler='invsqrt', **args)
            )
            valid2, test2 = testing_utils.train_model(
                dict(model_file=model_file, lr_scheduler='invsqrt', **args)
            )
            # make sure the number of updates is being tracked correctly
            self.assertGreater(
                valid2['total_train_updates'],
                valid1['total_train_updates'],
                'Number of updates is not increasing',
            )
            # make sure the learning rate is decreasing
            self.assertLess(
                valid2['lr'], valid1['lr'], 'Learning rate is not decreasing'
            )
            # but make sure we're not loading the scheduler if we're fine
            # tuning
            valid3, test3 = testing_utils.train_model(
                dict(
                    init_model=os.path.join(tmpdir, 'model'),
                    model_file=os.path.join(tmpdir, 'newmodel'),
                    lr_scheduler='invsqrt',
                    **args,
                )
            )
            self.assertEqual(
                valid3['total_train_updates'],
                valid1['total_train_updates'],
                'Finetuning LR scheduler reset failed (total_train_updates).',
            )
            self.assertEqual(
                valid3['lr'],
                valid1['lr'],
                'Finetuning LR scheduler reset failed ' '(lr).',
            )
            # and make sure we're not loading the scheduler if it changes
            valid4, test4 = testing_utils.train_model(
                dict(
                    init_model=os.path.join(tmpdir, 'model'),
                    model_file=os.path.join(tmpdir, 'newmodel2'),
                    lr_scheduler='reduceonplateau',
                    **args,
                )
            )
            self.assertEqual(
                valid4['total_train_updates'],
                valid1['total_train_updates'],
                'LR scheduler change reset failed (total_train_updates).',
            )
            self.assertEqual(
                valid4['lr'], 1e-3, '({}) LR is not correct in final resume.'
            )

    def test_resuming_generator(self):
        """
        Test generators resume correctly.
        """
        GENERATOR_ARGS = dict(
            task='integration_tests:nocandidate',
            model='transformer/generator',
            optimizer='sgd',
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
        self._test_learning_rate_resuming(GENERATOR_ARGS)

    def test_resuming_ranker(self):
        """
        Test resuming learning rate for the ranker.
        """
        RANKER_ARGS = dict(
            task='integration_tests:candidate',
            model='transformer/ranker',
            optimizer='sgd',
            learningrate=1e-3,
            batchsize=32,
            num_epochs=1,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=32,
            warmup_updates=1,
        )
        self._test_learning_rate_resuming(RANKER_ARGS)

    def test_invsqrt_learning_rate(self):
        args = dict(
            task='integration_tests:nocandidate',
            model='transformer/generator',
            learningrate=1,
            batchsize=1,
            warmup_updates=1,
            lr_scheduler='invsqrt',
            n_layers=1,
            n_heads=1,
            embedding_size=4,
            ffn_size=8,
            skip_generation=True,
            validation_max_exs=1,
            short_final_eval=True,
        )

        args['num_epochs'] = 9 / 500
        args['validation_every_n_epochs'] = 9 / 500
        valid1, test1 = testing_utils.train_model(args)
        args['num_epochs'] = 16 / 500
        args['validation_every_n_epochs'] = 16 / 500
        valid2, test2 = testing_utils.train_model(args)

        self.assertAlmostEqual(
            valid1['lr'],
            1 / 3,
            msg='Invsqrt LR {} was not 1/3 at step 9'.format(valid1['lr']),
            delta=0.001,
        )
        self.assertAlmostEqual(
            valid2['lr'],
            1 / 4,
            msg='Invsqrt LR {} was not 1/4 at step 16'.format(valid2['lr']),
            delta=0.001,
        )


if __name__ == '__main__':
    unittest.main()

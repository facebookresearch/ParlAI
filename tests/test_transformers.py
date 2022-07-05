#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test many variants of transformers.
"""

import os
import torch
import unittest
from unittest.mock import MagicMock
import pytest
import parlai.utils.testing as testing_utils
from parlai.agents.transformer.modules import (
    TransformerFFN,
    TransformerGeneratorModel,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from parlai.core.agents import create_agent
from parlai.core.agents import create_agent_from_model_file
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from .test_dict import DEFAULT_BYTELEVEL_BPE_VOCAB, DEFAULT_BYTELEVEL_BPE_MERGE
from parlai.core.params import ParlaiParser


class TestTransformerBase(unittest.TestCase):
    """
    Base Tester class for sharing functionality.
    """

    def _test_resize_embeddings(self, model):
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model_file')
            _, _ = testing_utils.train_model(
                dict(
                    model=model,
                    task='integration_tests:short_fixed',
                    n_layers=1,
                    n_encoder_layers=2,
                    n_decoder_layers=4,
                    num_epochs=1,
                    dict_tokenizer='bytelevelbpe',
                    bpe_vocab=DEFAULT_BYTELEVEL_BPE_VOCAB,
                    bpe_merge=DEFAULT_BYTELEVEL_BPE_MERGE,
                    bpe_add_prefix_space=False,
                    model_file=model_file,
                    save_after_valid=True,
                )
            )

            # now create agent with special tokens
            parser = ParlaiParser()
            parser.set_params(
                model=model,
                task='integration_tests:short_fixed',
                n_layers=1,
                n_encoder_layers=2,
                n_decoder_layers=4,
                dict_tokenizer='bytelevelbpe',
                bpe_vocab=DEFAULT_BYTELEVEL_BPE_VOCAB,
                bpe_merge=DEFAULT_BYTELEVEL_BPE_MERGE,
                bpe_add_prefix_space=False,
                model_file=model_file,
                save_after_valid=True,
                special_tok_lst='PARTY,PARROT',
            )
            opt = parser.parse_args([])
            agent = create_agent(opt)
            # assert that the embeddings were resized
            assert agent.resized_embeddings
            # assert model has special tokens
            self.assertEqual(agent.special_toks, ['PARTY', 'PARROT'])


class TestTransformerRanker(unittest.TestCase):
    """
    Checks that transformer_ranker can learn some very basic tasks.
    """

    def _overfit_train(self, **args):
        opt = dict(
            task='integration_tests:overfit',
            model='transformer/ranker',
            optimizer='adam',
            learningrate=0.01,
            batchsize=4,
            validation_every_n_epochs=5,
            validation_patience=10,
            lr_scheduler='none',
            n_layers=1,
            n_heads=4,
            ffn_size=64,
            embedding_size=8,
            candidates='batch',
            eval_candidates='batch',
            gradient_clip=0.5,
        )
        opt.update(args)
        return testing_utils.train_model(opt)

    @testing_utils.retry(ntries=3)
    def test_repeater(self):
        """
        Test a simple repeat-after-me model.
        """
        valid, test = self._overfit_train()

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
                    batchsize=16,
                    num_epochs=0.1,
                    n_layers=1,
                    n_heads=1,
                    ffn_size=4,
                    embedding_size=4,
                    warmup_updates=0,
                    lr_scheduler='invsqrt',
                )
            )

            valid2, test2 = testing_utils.train_model(
                dict(
                    model_file=model_file,
                    task='integration_tests:candidate',
                    model='transformer/ranker',
                    num_epochs=0.1,
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
                    warmup_updates=0,
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
                batchsize=64,
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
        valid, test = self._overfit_train(variant='xlm', activation='gelu')

        self.assertGreaterEqual(valid['hits@1'], 0.90)
        self.assertGreaterEqual(test['hits@1'], 0.90)

    @testing_utils.retry(ntries=3)
    def test_prelayernorm(self):
        """
        Test --variant prelayernorm with history_add_global_end_token option.
        """
        valid, test = self._overfit_train(
            task='integration_tests:overfit',
            model='transformer/ranker',
            variant='prelayernorm',
            activation='gelu',
            history_add_global_end_token='end',
        )

        self.assertGreaterEqual(valid['hits@1'], 0.90)
        self.assertGreaterEqual(test['hits@1'], 0.90)

    @testing_utils.retry(ntries=3)
    def test_alt_reduction(self):
        """
        Test a transformer ranker reduction method other than `mean`.
        """
        valid, test = self._overfit_train(
            variant='xlm',
            activation='gelu',
            reduction_type='first',  # this is really what we're trying to test for
        )

        self.assertGreaterEqual(valid['hits@1'], 0.99)
        self.assertGreaterEqual(test['hits@1'], 0.99)


class TestTransformerGenerator(TestTransformerBase):
    """
    Checks that the generative transformer can learn basic tasks.
    """

    def _overfit_train(self, **args):
        args = dict(
            task='integration_tests:overfit',
            model='transformer/generator',
            optimizer='sgd',
            learningrate=1,
            momentum=0.9,
            batchsize=4,
            n_layers=1,
            n_heads=1,
            ffn_size=32,
            embedding_size=16,
            inference='greedy',
            beam_size=1,
            skip_generation=True,
            validation_metric='ppl',
            validation_every_n_epochs=10,
            num_epochs=100,
        )
        args.update(args)
        return testing_utils.train_model(args)

    def test_checkpoint(self):
        """
        Checks --checkpoint-activations true.
        """
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:overfit',
                model='transformer/generator',
                dict_file='zoo:unittest/transformer_generator2/model.dict',
                batchsize=4,
                skip_generation=True,
                validation_metric='ppl',
                max_train_steps=10,
                checkpoint_activations=True,
            )
        )

    def test_greedysearch(self):
        """
        Test greedy search.
        """
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='transformer/generator',
                model_file='zoo:unittest/transformer_generator2/model',
                batchsize=4,
                inference='greedy',
                metrics='bleu',
                beam_size=1,
                num_examples=20,
            )
        )

        self.assertLessEqual(valid['ppl'], 1.05)
        # 0.75 because some of the turns contain fewer than 2 words
        self.assertAlmostEqual(valid['bleu-2'], 0.60, delta=0.001)
        self.assertAlmostEqual(valid['bleu-3'], 0.40, delta=0.001)
        self.assertLessEqual(test['ppl'], 1.05)
        self.assertAlmostEqual(test['bleu-2'], 0.60, delta=0.001)
        self.assertAlmostEqual(test['bleu-3'], 0.40, delta=0.001)

    def test_beamsearch(self):
        """
        Test beamsearch.
        """
        valid, test = testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='transformer/generator',
                model_file='zoo:unittest/transformer_generator2/model',
                batchsize=4,
                metrics='bleu',
                inference='beam',
                beam_size=5,
                num_examples=20,
            )
        )

        self.assertLessEqual(valid['ppl'], 1.05)
        self.assertAlmostEqual(valid['bleu-2'], 0.60, delta=0.001)
        self.assertAlmostEqual(valid['bleu-3'], 0.40, delta=0.001)
        self.assertLessEqual(test['ppl'], 1.05)
        self.assertAlmostEqual(test['bleu-2'], 0.60, delta=0.001)
        self.assertAlmostEqual(test['bleu-3'], 0.40, delta=0.001)

    @pytest.mark.nofbcode
    def test_beamsearch_return_all_texts(self):
        """
        Test beam_texts for beam_size > 1.
        """
        size = 3

        agent = create_agent_from_model_file(
            'zoo:unittest/beam_blocking/model',
            opt_overrides={"beam_size": size, "inference": "beam"},
        )
        agent.observe({'text': '5 5 5 5 5 5 5', 'episode_done': True})
        response = agent.act()
        self.assertTrue("beam_texts" in response)
        self.assertGreaterEqual(len(response["beam_texts"]), size)
        hyp, score = response["beam_texts"][0]
        self.assertTrue(isinstance(hyp, str))
        self.assertTrue(isinstance(score, float))

        agent = create_agent_from_model_file(
            'zoo:unittest/beam_blocking/model',
            opt_overrides={"beam_size": size, "inference": "topk"},
        )
        agent.observe({'text': '5 5 5 5 5 5 5', 'episode_done': True})
        response = agent.act()
        self.assertTrue("beam_texts" in response)
        self.assertEqual(len(response["beam_texts"]), size)

    @pytest.mark.nofbcode
    def test_beamsearch_blocking_cpu(self):
        """
        Test beamsearch blocking.
        """
        with testing_utils.tempdir() as tmpdir:
            agent = create_agent_from_model_file('zoo:unittest/beam_blocking/model')
            agent.observe({'text': '5 5 5 5 5 5 5', 'episode_done': True})
            assert agent.act()['text'] == '5 5 5 5 5 5 5'

            agent = create_agent_from_model_file(
                'zoo:unittest/beam_blocking/model', Opt(beam_block_ngram=1)
            )
            agent.observe({'text': '5 5 5 5 5 5 5', 'episode_done': True})
            assert '5 5' not in agent.act()['text']

            agent = create_agent_from_model_file(
                'zoo:unittest/beam_blocking/model', Opt(beam_block_ngram=2)
            )
            agent.observe({'text': '5 5 5 5 5 5 5', 'episode_done': True})
            assert '5 5 5' not in agent.act()['text']

            with open(os.path.join(tmpdir, 'blocklist.txt'), 'w') as f:
                f.write("38\n62\n34 34\n")

            agent = create_agent_from_model_file(
                'zoo:unittest/beam_blocking/model',
                Opt(beam_block_list_filename=os.path.join(tmpdir, 'blocklist.txt')),
            )
            agent.observe({'text': '4 4 4', 'episode_done': True})
            assert agent.act()['text'] == '4 4 4'

            agent.observe({'text': '38 38 38', 'episode_done': True})
            assert '38' not in agent.act()['text']

            agent.observe({'text': '62 62 62', 'episode_done': True})
            assert '62' not in agent.act()['text']

            agent.observe({'text': '34 34 34', 'episode_done': True})
            text = agent.act()['text']
            assert '34' in text
            assert '34 34' not in text

    @pytest.mark.nofbcode
    @testing_utils.skipUnlessGPU
    def test_beamsearch_blocking_gpu(self):
        """
        Test beamsearch blocking.
        """
        with testing_utils.tempdir() as tmpdir:
            agent = create_agent_from_model_file(
                'zoo:unittest/beam_blocking/model',
                Opt(gpu_beam_blocking=True),
            )
            agent.observe({'text': '5 5 5 5 5 5 5', 'episode_done': True})
            assert agent.act()['text'] == '5 5 5 5 5 5 5'

            agent = create_agent_from_model_file(
                'zoo:unittest/beam_blocking/model',
                Opt(beam_block_ngram=1, gpu_beam_blocking=True),
            )
            agent.observe({'text': '5 5 5 5 5 5 5', 'episode_done': True})
            assert '5 5' not in agent.act()['text']

            agent = create_agent_from_model_file(
                'zoo:unittest/beam_blocking/model',
                Opt(beam_block_ngram=2, gpu_beam_blocking=True),
            )
            agent.observe({'text': '5 5 5 5 5 5 5', 'episode_done': True})
            assert '5 5 5' not in agent.act()['text']

            with open(os.path.join(tmpdir, 'blocklist.txt'), 'w') as f:
                f.write("38\n62\n34 34\n")

            agent = create_agent_from_model_file(
                'zoo:unittest/beam_blocking/model',
                Opt(
                    beam_block_list_filename=os.path.join(tmpdir, 'blocklist.txt'),
                    gpu_beam_blocking=True,
                ),
            )
            agent.observe({'text': '4 4 4', 'episode_done': True})
            assert agent.act()['text'] == '4 4 4'

            agent.observe({'text': '38 38 38', 'episode_done': True})
            assert '38' not in agent.act()['text']

            agent.observe({'text': '62 62 62', 'episode_done': True})
            assert '62' not in agent.act()['text']

            agent.observe({'text': '34 34 34', 'episode_done': True})
            text = agent.act()['text']
            assert '34' in text
            assert '34 34' not in text

    @pytest.mark.nofbcode
    def test_beamsearch_contextblocking_cpu(self):
        """
        Test beamsearch context blocking.
        """

        agent = create_agent_from_model_file('zoo:unittest/context_blocking/model')
        agent.observe({'text': '5 4 3 2', 'episode_done': True})
        assert agent.act()['text'] == '5 4 3 2'

        agent = create_agent_from_model_file(
            'zoo:unittest/context_blocking/model', Opt(beam_context_block_ngram=1)
        )
        agent.observe({'text': '5 4 3 2', 'episode_done': True})
        text = agent.act()['text']
        assert '5' not in text
        assert '4' not in text
        assert '3' not in text
        assert '2' not in text

        agent = create_agent_from_model_file(
            'zoo:unittest/context_blocking/model', Opt(beam_context_block_ngram=2)
        )
        agent.observe({'text': '5 4 3 2', 'episode_done': True})
        text = agent.act()['text']
        assert '5' in text
        assert '5 4' not in text
        assert '4 3' not in text
        assert '3 2' not in text

    @pytest.mark.nofbcode
    @testing_utils.skipUnlessGPU
    def test_beamsearch_contextblocking_gpu(self):
        """
        Test beamsearch context blocking.
        """

        agent = create_agent_from_model_file(
            'zoo:unittest/context_blocking/model',
            Opt(gpu_beam_blocking=True),
        )
        agent.observe({'text': '5 4 3 2', 'episode_done': True})
        assert agent.act()['text'] == '5 4 3 2'

        agent = create_agent_from_model_file(
            'zoo:unittest/context_blocking/model',
            Opt(beam_context_block_ngram=1, gpu_beam_blocking=True),
        )
        agent.observe({'text': '5 4 3 2', 'episode_done': True})
        text = agent.act()['text']
        assert '5' not in text
        assert '4' not in text
        assert '3' not in text
        assert '2' not in text

        agent = create_agent_from_model_file(
            'zoo:unittest/context_blocking/model',
            Opt(beam_context_block_ngram=2, gpu_beam_blocking=True),
        )
        agent.observe({'text': '5 4 3 2', 'episode_done': True})
        text = agent.act()['text']
        assert '5' in text
        assert '5 4' not in text
        assert '4 3' not in text
        assert '3 2' not in text

    def test_nucleus(self):
        """
        Test nucleus generation.
        """
        # Nucleus is inherently stochastic, just ensure no crash.
        opt = ParlaiParser(True, True).parse_kwargs(
            model_file='zoo:unittest/transformer_generator2/model',
            inference='nucleus',
            topp=0.3,
        )
        agent = create_agent(opt, True)
        agent.observe({'text': '1', 'episode_done': True})
        result = agent.act()
        assert 'text' in result
        assert result['text'] != ''

    def test_beamdelay(self):
        """
        Test delayedbeam generation.
        """
        # Delayed Beam is inherently stochastic, just ensure no crash.
        opt = ParlaiParser(True, True).parse_kwargs(
            model_file='zoo:unittest/transformer_generator2/model',
            inference='delayedbeam',
            topk=10,
            beam_delay=2,
            beam_min_length=2,
        )
        agent = create_agent(opt, True)
        agent.observe({'text': '1\n1\n2\n2\n3\n3\n4', 'episode_done': True})
        result = agent.act()
        assert 'text' in result
        assert result['text'] != ''
        assert '1 2' in result['text']

    def test_topk(self):
        """
        Test topk generation.
        """
        # Topk is inherently stochastic, just ensure no crash.
        opt = ParlaiParser(True, True).parse_kwargs(
            model_file='zoo:unittest/transformer_generator2/model',
            inference='topk',
            topp=10,
        )
        agent = create_agent(opt, True)
        agent.observe({'text': '1', 'episode_done': True})
        result = agent.act()
        assert 'text' in result
        assert result['text'] != ''

    def test_generator_backcomp(self):
        """
        Tests that the generator model files work over time.
        """
        _, test = testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='transformer/generator',
                model_file='zoo:unittest/transformer_generator2/model',
                dict_file='zoo:unittest/transformer_generator2/model.dict',
                rank_candidates=False,
                batchsize=64,
            ),
            skip_valid=True,
        )

        self.assertLessEqual(test['ppl'], 1.01)
        self.assertGreaterEqual(test['accuracy'], 0.99)
        self.assertGreaterEqual(test['f1'], 0.99)

    @testing_utils.retry(ntries=3)
    def test_xlm(self):
        """
        Test --variant xlm.
        """
        valid, test = self._overfit_train(
            variant='xlm',
            activation='gelu',
            n_segments=8,  # doesn't do anything but still good to test
            adam_eps=1e-6,  # just to test another flag simultaneously
        )

        self.assertLessEqual(valid['ppl'], 1.02)
        self.assertLessEqual(test['ppl'], 1.02)

    @testing_utils.retry(ntries=3)
    def test_prelayernorm(self):
        """
        Test --variant prelayernorm.
        """
        valid, test = self._overfit_train(variant='prelayernorm', activation='gelu')

        self.assertLessEqual(valid['ppl'], 1.30)
        self.assertLessEqual(test['ppl'], 1.30)

    @pytest.mark.nofbcode
    def test_compute_tokenized_bleu(self):
        """
        Test that the model outputs self-computed bleu correctly.
        """
        valid, _ = testing_utils.eval_model(
            dict(
                task='integration_tests',
                model_file='zoo:unittest/context_blocking/model',
                dict_file='zoo:unittest/context_blocking/model.dict',
                inference='greedy',
                beam_size=1,
                skip_generation=False,
                compute_tokenized_bleu=True,
                metrics='all',
            )
        )
        try:
            import fairseq  # @manual # noqa: F401

            assert valid['fairseq_bleu1'] > 0.9
        except ImportError:
            # fairseq not installed, let's just move on
            pass

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

    def test_temperature(self):
        """
        Test temperature.
        """
        # Just ensuring no crash.
        testing_utils.eval_model(
            dict(
                task='integration_tests:multiturn_candidate',
                model='transformer/generator',
                model_file='zoo:unittest/transformer_generator2/model',
                batchsize=32,
                inference='beam',
                beam_size=5,
                temperature=0.99,
            )
        )

    @pytest.mark.nofbcode
    def test_resize_embeddings(self):
        self._test_resize_embeddings('transformer/generator')


class TestClassifier(unittest.TestCase):
    """
    Test transformer/classifier.
    """

    @testing_utils.retry()
    def test_simple(self):
        valid, test = testing_utils.train_model(
            dict(
                task='integration_tests:classifier',
                model='transformer/classifier',
                classes=['one', 'zero'],
                optimizer='adamax',
                truncate=8,
                learningrate=7e-3,
                batchsize=32,
                num_epochs=5,
                n_layers=1,
                n_heads=1,
                ffn_size=32,
                embedding_size=32,
            )
        )
        assert valid['accuracy'] > 0.97
        assert test['accuracy'] > 0.97


class TestLearningRateScheduler(unittest.TestCase):
    """
    Test learning rate scheduler for both generative and ranking transformers.
    """

    def _test_learning_rate_resuming(self, user_args):
        """
        Test learning rate resumes correctly.
        """
        args = dict(
            task='integration_tests:overfit',
            lr_scheduler='invsqrt',
            optimizer='sgd',
            learningrate=1e-3,
            batchsize=4,
            num_epochs=1,
            n_layers=1,
            n_heads=1,
            ffn_size=4,
            embedding_size=4,
        )
        args.update(user_args)

        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model')
            args['model_file'] = model_file
            valid1, test1 = testing_utils.train_model(args)
            valid2, test2 = testing_utils.train_model(args)
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

            del args['lr_scheduler']
            del args['model_file']
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
                valid3['lr'], valid1['lr'], 'Finetuning LR scheduler reset failed (lr).'
            )
            # and make sure we're not loading the scheduler if it changes
            valid4, test4 = testing_utils.train_model(
                dict(
                    init_model=os.path.join(tmpdir, 'model'),
                    model_file=os.path.join(tmpdir, 'newmodel2'),
                    lr_scheduler='reduceonplateau',
                    log_every_n_secs=0.001,
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
            model='transformer/generator', skip_generation=True, warmup_updates=0
        )
        self._test_learning_rate_resuming(GENERATOR_ARGS)

    def test_resuming_ranker(self):
        """
        Test resuming learning rate for the ranker.
        """
        RANKER_ARGS = dict(model='transformer/ranker', warmup_updates=0)
        self._test_learning_rate_resuming(RANKER_ARGS)

    def test_invsqrt_learning_rate(self):
        args = dict(
            task='integration_tests:nocandidate',
            model='transformer/generator',
            learningrate=1,
            batchsize=1,
            warmup_updates=0,
            lr_scheduler='invsqrt',
            n_layers=1,
            n_heads=1,
            embedding_size=4,
            ffn_size=8,
            skip_generation=True,
            validation_max_exs=1,
            short_final_eval=True,
        )

        args['num_epochs'] = args['validation_every_n_epochs'] = 9 / 500
        valid1, test1 = testing_utils.train_model(args)
        args['num_epochs'] = args['validation_every_n_epochs'] = 16 / 500
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


class TestPolyencoder(TestTransformerBase):
    """
    Unit tests for PolyencoderAgent.
    """

    @pytest.mark.nofbcode
    def test_resize_embeddings(self):
        self._test_resize_embeddings('transformer/polyencoder')

    def test_multi_head_attention(self):
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model_file')
            _, _ = testing_utils.train_model(
                Opt(
                    model='transformer/polyencoder',
                    task='integration_tests:short_fixed',
                    n_layers=1,
                    n_encoder_layers=2,
                    n_decoder_layers=4,
                    num_epochs=1,
                    dict_tokenizer='bytelevelbpe',
                    bpe_vocab=DEFAULT_BYTELEVEL_BPE_VOCAB,
                    bpe_merge=DEFAULT_BYTELEVEL_BPE_MERGE,
                    bpe_add_prefix_space=False,
                    model_file=model_file,
                    save_after_valid=True,
                    poly_attention_type='multihead',
                    codes_attention_type='multihead',
                )
            )


@testing_utils.skipUnlessVision
class TestImagePolyencoder(unittest.TestCase):
    """
    Unit tests for the ImagePolyencoderAgent.

    Test that the model is able to handle simple train tasks.
    """

    base_args = {
        'log_every_n_secs': 5,
        'model': 'transformer/image_polyencoder',
        'embedding_size': 32,
        'n_heads': 1,
        'n_layers': 1,
        'n_positions': 128,
        'truncate': 128,
        'ffn_size': 32,
        'variant': 'xlm',
        'activation': 'gelu',
        'candidates': 'batch',
        'eval_candidates': 'batch',  # No inline cands
        'embeddings_scale': False,
        'gradient_clip': -1.0,
        'learningrate': 1e-4,
        'batchsize': 8,
        'optimizer': 'adamax',
        'learn_positional_embeddings': False,
        'reduction_type': 'mean',
        'num_epochs': 10,
    }
    text_args = {'task': 'integration_tests:nocandidate'}
    image_args = {
        'task': 'integration_tests:ImageTeacher',
        'image_mode': 'resnet152',
        'image_features_dim': 2048,
        'image_encoder_num_layers': 1,
        'image_combination_mode': 'prepend',
        'n_image_tokens': 1,
        'num_epochs': 20,
    }
    multitask_args = {
        'task': 'integration_tests:nocandidate,integration_tests:ImageTeacher',
        'image_mode': 'resnet152',
        'image_features_dim': 2048,
        'image_encoder_num_layers': 1,
        'image_combination_mode': 'prepend',
        'n_image_tokens': 1,
        'multitask_weights': [1, 1],
        'num_epochs': 30,
    }

    @testing_utils.retry(ntries=3)
    def test_text_task(self):
        """
        Test that model correctly handles text task.

        Random chance is 10%, so this should be able to get much better than that very
        quickly.
        """
        args = Opt({**self.base_args, **self.text_args})
        valid, test = testing_utils.train_model(args)
        assert (
            valid['accuracy'] > 0.1
        ), f'ImagePolyencoderAgent val-set accuracy on a simple task was {valid["accuracy"].value():0.2f}.'

    @testing_utils.retry(ntries=3)
    @testing_utils.skipUnlessTorch
    @testing_utils.skipUnlessGPU
    def test_image_task(self):
        """
        Test that model correctly handles a basic image training task.

        Random chance is 10%, so this should be able to get much better than that very
        quickly.
        """
        args = Opt({**self.base_args, **self.image_args})
        valid, test = testing_utils.train_model(args)
        assert (
            valid['accuracy'] > 0.05
        ), f'ImagePolyencoderAgent val-set accuracy on a simple task was {valid["accuracy"].value():0.2f}.'

    @testing_utils.retry(ntries=3)
    @testing_utils.skipUnlessTorch
    @testing_utils.skipUnlessGPU
    def test_multitask(self):
        """
        Test that model correctly handles multiple inputs.

        Random chance is 10%, so this should be able to get much better than that very
        quickly.
        """
        args = Opt({**self.base_args, **self.multitask_args})
        valid, test = testing_utils.train_model(args)
        assert (
            valid['accuracy'] > 0.1
        ), f'ImagePolyencoderAgent val-set accuracy on a simple task was {valid["accuracy"].value():0.2f}.'


class TestSwappableComponents(unittest.TestCase):
    def _opt(self, **kwargs):
        return Opt(
            batchsize=4,
            optimizer='adam',
            n_layers=1,
            n_heads=4,
            ffn_size=16,
            embedding_size=16,
            skip_generation=True,
            **kwargs,
        )

    def test_swap_encoder_attention(self):
        CustomFFN = type('CustomFFN', (TransformerFFN,), {})
        CustomFFN.forward = MagicMock()
        wrapped_class = TransformerGeneratorModel.with_components(
            encoder=TransformerEncoder.with_components(
                layer=TransformerEncoderLayer.with_components(feedforward=CustomFFN)
            )
        )
        opt = self._opt()
        CustomFFN.forward.assert_not_called
        model = wrapped_class(opt=opt, dictionary=DictionaryAgent(opt))
        assert isinstance(model, TransformerGeneratorModel)  # type: ignore
        try:
            model(torch.zeros(1, 1).long(), ys=torch.zeros(1, 1).long())  # type: ignore
        except TypeError:
            pass
        finally:
            CustomFFN.forward.assert_called

    def test_swap_is_not_persisted_in_class(self):
        opt = self._opt()
        dictionary = DictionaryAgent(opt)

        CustomFFN = type('CustomFFN', (TransformerFFN,), {})
        wrapped_class = TransformerGeneratorModel.with_components(
            encoder=TransformerEncoder.with_components(
                layer=TransformerEncoderLayer.with_components(feedforward=CustomFFN)
            )
        )
        model = wrapped_class(opt=opt, dictionary=dictionary)
        assert (
            model.swappables.encoder.swappables.layer.swappables.feedforward
            == CustomFFN
        )  # type: ignore

        another_model = TransformerGeneratorModel(opt, dictionary)
        assert another_model.swappables != model.swappables
        assert issubclass(
            another_model.swappables.encoder, TransformerEncoder
        )  # type: ignore

        wrapped_class.swap_components(
            encoder=TransformerEncoder.with_components(
                layer=TransformerEncoderLayer.with_components(
                    feedforward=TransformerFFN
                )
            )
        )
        one_more_model = wrapped_class(opt=opt, dictionary=dictionary)
        assert (
            one_more_model.swappables.encoder.swappables.layer.swappables.feedforward
            == TransformerFFN
        )  # type: ignore

    def test_examples_variant(self):
        opt = ParlaiParser(True, True).parse_kwargs(
            model='parlai.agents.examples.transformer_variant:TransformerVariantAgent'
        )
        model = create_agent(opt)
        # send the model a single training example to ensure it can forward/backward
        model.observe({'text': '1 2 3 4', 'labels': ['1 2 3 4'], 'episode_done': True})
        model.act()
        # send the model a single validation example
        model.observe(
            {'text': '1 2 3 4', 'eval_labels': ['1 2 3 4'], 'episode_done': True}
        )
        model.act()

    def test_examples_configurable(self):
        opt = ParlaiParser(True, True).parse_kwargs(
            model='parlai.agents.examples.transformer_variant:ConfigurableTransformerAgent',
            decoder_ffn_variants='two',
        )
        model = create_agent(opt)
        # send the model a single training example to ensure it can forward/backward
        model.observe({'text': '1 2 3 4', 'labels': ['1 2 3 4'], 'episode_done': True})
        model.act()
        # send the model a single validation example
        model.observe(
            {'text': '1 2 3 4', 'eval_labels': ['1 2 3 4'], 'episode_done': True}
        )
        model.act()


class TestDecoderOnly(unittest.TestCase):
    """
    Unit tests for DecoderOnlyAgent.
    """

    @pytest.mark.nofbcode
    def test_resize_embeddings(self):
        model = 'transformer/decoder'
        with testing_utils.tempdir() as tmpdir:
            model_file = os.path.join(tmpdir, 'model_file')
            _, _ = testing_utils.train_model(
                dict(
                    model=model,
                    task='integration_tests:short_fixed',
                    n_layers=2,
                    num_epochs=1,
                    dict_tokenizer='bytelevelbpe',
                    bpe_vocab=DEFAULT_BYTELEVEL_BPE_VOCAB,
                    bpe_merge=DEFAULT_BYTELEVEL_BPE_MERGE,
                    bpe_add_prefix_space=False,
                    model_file=model_file,
                    save_after_valid=True,
                )
            )

            # now create agent with special tokens
            parser = ParlaiParser()
            parser.set_params(
                model=model,
                task='integration_tests:short_fixed',
                n_layers=2,
                dict_tokenizer='bytelevelbpe',
                bpe_vocab=DEFAULT_BYTELEVEL_BPE_VOCAB,
                bpe_merge=DEFAULT_BYTELEVEL_BPE_MERGE,
                bpe_add_prefix_space=False,
                model_file=model_file,
                save_after_valid=True,
                special_tok_lst='PARTY,PARROT',
            )
            opt = parser.parse_args([])
            agent = create_agent(opt)
            # assert that the embeddings were resized
            assert agent.resized_embeddings
            # assert model has special tokens
            self.assertEqual(agent.special_toks, ['PARTY', 'PARROT'])

    def _overfit_train(self, **args):
        args = dict(
            task='integration_tests:overfit',
            model='transformer/decoder',
            optimizer='sgd',
            learningrate=1,
            momentum=0.9,
            batchsize=4,
            n_layers=2,
            n_heads=1,
            ffn_size=32,
            embedding_size=16,
            inference='greedy',
            beam_size=1,
            skip_generation=True,
            validation_metric='ppl',
            validation_every_n_epochs=10,
            num_epochs=100,
        )
        args.update(args)
        return testing_utils.train_model(args)

    @testing_utils.retry(ntries=3)
    def test_train(self):
        """
        Test basic training.
        """
        valid, test = self._overfit_train(variant='prelayernorm', activation='gelu')

        self.assertLessEqual(valid['ppl'], 1.30)
        self.assertLessEqual(test['ppl'], 1.30)


if __name__ == '__main__':
    unittest.main()

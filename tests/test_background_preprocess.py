#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for QueueWorlds.
"""
import unittest
import parlai.utils.testing as testing_utils
from parlai.tasks.integration_tests.agents import NUM_TEST

BASE_ARGS = {
    'model': 'transformer/generator',
    'embedding_size': 16,
    'n_heads': 1,
    'n_layers': 1,
    'n_positions': 32,
    'truncate': 32,
    'ffn_size': 64,
    'variant': 'xlm',
    'activation': 'gelu',
    'embeddings_scale': True,
    'gradient_clip': 0.1,
    # Train args
    'learningrate': 7e-3,
    'optimizer': 'adamax',
    'learn_positional_embeddings': True,
    'num_workers': 2,
}

SINGLETASK_ARGS = {'task': 'integration_tests:nocandidate', 'num_epochs': 0.1}

SINGLETASK_STREAM_ARGS = {'task': 'integration_tests:no_candidate_fb_dialog', 'num_epochs': 0.1}

SINGLETASK_MULTIVALID_ARGS = {
    'task': 'integration_tests:nocandidate',
    'num_epochs': 0.2,
    'validation_every_n_epochs': 0.1,
    'validation_metric': 'ppl',
}

MULTITASK_ARGS = {
    'task': 'integration_tests:nocandidate,integration_tests:multiturn_nocandidate',
    'num_epochs': 0.1,
}

MULTITASK_MULTIVALID_ARGS = {
    'task': 'integration_tests:nocandidate,integration_tests:multiturn_nocandidate',
    'num_epochs': 0.2,
    'validation_every_n_epochs': 0.1,
    'validation_metric': 'ppl',
}

MULTIPROCESS_ARGS = {
    'task': 'integration_tests:no_candidate_fb_dialog',
    'num_epochs': 1,
    'batchsize': 16,
}


class TestBackgroundPreprocess(unittest.TestCase):
    """
    Test the Q worlds and the P worlds.
    """

    def test_singletask(self):
        """
        Normal test - test if model can train via batchworld.
        """
        for extra_args in [SINGLETASK_ARGS, SINGLETASK_MULTIVALID_ARGS]:
            for bsz in [1, 16]:
                short_final_eval = bsz == 1
                args = BASE_ARGS.copy()
                args.update(extra_args)
                args['batchsize'] = bsz
                args['short_final_eval'] = short_final_eval
                vme_multiplier = 0.1 if short_final_eval else 1
                args['validation_max_exs'] = int(NUM_TEST * vme_multiplier)
                valid, test = testing_utils.train_model(args)
                for report in [valid, test]:
                    self.assertEqual(
                        report['exs'], NUM_TEST * vme_multiplier, f'args: {args}'
                    )
                self.assertEqual(
                    valid['total_train_updates'], test['total_train_updates']
                )

    def test_singletask_dynb(self):
        """
        Test batchworld singletask.
        """
        for extra_args in [SINGLETASK_ARGS, SINGLETASK_MULTIVALID_ARGS]:
            short_final_eval = extra_args == SINGLETASK_MULTIVALID_ARGS
            vme_multiplier = 0.1 if short_final_eval else 1
            args = BASE_ARGS.copy()
            args.update(extra_args)
            args['batchsize'] = 8
            args['dynamic_batching'] = 'full'
            args['short_final_eval'] = short_final_eval
            args['validation_max_exs'] = NUM_TEST * vme_multiplier
            valid, test = testing_utils.train_model(args)
            for report in [valid, test]:
                if short_final_eval:
                    # dynbatching behaves strangely with vme != num total eval exs
                    self.assertLess(report['exs'], NUM_TEST, f'args {args}')
                else:
                    self.assertEqual(report['exs'], NUM_TEST, f'args: {args}')
            self.assertEqual(valid['total_train_updates'], test['total_train_updates'])

    def test_multitask(self):
        """
        Normal test - test if model can train via batchworld.
        """
        for extra_args in [MULTITASK_ARGS, MULTITASK_MULTIVALID_ARGS]:
            for bsz in [1, 16]:
                short_final_eval = bsz == 1
                args = BASE_ARGS.copy()
                args.update(extra_args)
                args['batchsize'] = bsz
                vme_multiplier = 0.1 if short_final_eval else 1
                args['short_final_eval'] = True if short_final_eval else False
                args['validation_max_exs'] = int(NUM_TEST * vme_multiplier)
                valid, test = testing_utils.train_model(args)
                for rep in [valid, test]:
                    num_no_cand = (
                        (NUM_TEST * vme_multiplier) / 2
                        if short_final_eval
                        else NUM_TEST
                    )
                    num_multiturn = (
                        NUM_TEST * 4 if not short_final_eval else num_no_cand
                    )
                    self.assertEqual(
                        rep['exs'], num_no_cand + num_multiturn, f'args: {args}'
                    )
                    self.assertEqual(
                        rep['integration_tests:nocandidate/exs'],
                        num_no_cand,
                        f'args: {args}',
                    )
                    self.assertEqual(
                        rep['integration_tests:multiturn_nocandidate/exs'],
                        num_multiturn,
                        f'args: {args}',
                    )

                self.assertEqual(
                    valid['total_train_updates'],
                    test['total_train_updates'],
                    f'args: {args}',
                )

    def test_multitask_dynamic_batching(self):
        """
        Multitask dynamic batching.
        """
        for extra_args in [MULTITASK_ARGS, MULTITASK_MULTIVALID_ARGS]:
            short_final_eval = extra_args == MULTITASK_MULTIVALID_ARGS
            vme_multiplier = 0.05 if short_final_eval else 1
            args = BASE_ARGS.copy()
            args.update(extra_args)
            args['batchsize'] = 8
            args['dynamic_batching'] = 'full'
            args['short_final_eval'] = short_final_eval
            args['validation_max_exs'] = NUM_TEST * vme_multiplier
            valid, test = testing_utils.train_model(args)
            for rep in [valid, test]:
                assertion = self.assertLess if short_final_eval else self.assertEqual
                assertion(rep['exs'], NUM_TEST + NUM_TEST * 4, f'args: {args}')
                assertion(
                    rep['integration_tests:nocandidate/exs'], NUM_TEST, f'args: {args}'
                )
                assertion(
                    rep['integration_tests:multiturn_nocandidate/exs'],
                    NUM_TEST * 4,
                    f'args: {args}',
                )

            self.assertEqual(valid['total_train_updates'], test['total_train_updates'])

    def test_stream(self):
        """
        Test Streaming.
        """
        args = BASE_ARGS.copy()
        args.update(SINGLETASK_STREAM_ARGS)
        args['batchsize'] = 8
        args['datatype'] = 'train:stream'
        valid, test = testing_utils.train_model(args)
        for report in [valid, test]:
            self.assertEqual(report['exs'], NUM_TEST, f'args: {args}')
        self.assertEqual(valid['total_train_updates'], test['total_train_updates'])

    @testing_utils.skipUnlessGPU
    def test_singletask_distributed(self):
        """
        Distributed Training.
        """
        args = BASE_ARGS.copy()
        args.update(MULTIPROCESS_ARGS)
        for dt in ['train', 'train:stream']:
            args['datatype'] = dt
            valid, test = testing_utils.distributed_train_model(args)
            for report in [valid, test]:
                self.assertEqual(report['exs'], NUM_TEST, f'args: {args}')
            self.assertEqual(valid['total_train_updates'], test['total_train_updates'])


if __name__ == "__main__":
    unittest.main()

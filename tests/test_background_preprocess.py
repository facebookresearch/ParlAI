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
    'embedding_size': 32,
    'n_heads': 2,
    'n_layers': 2,
    'n_positions': 128,
    'truncate': 128,
    'ffn_size': 128,
    'variant': 'xlm',
    'activation': 'gelu',
    'embeddings_scale': True,
    'gradient_clip': 0.1,
    # Train args
    'learningrate': 7e-3,
    'batchsize': 16,
    'optimizer': 'adamax',
    'learn_positional_embeddings': True,
    'numworkers': 2,
}

SINGLETASK_ARGS = {'task': 'integration_tests:nocandidate', 'num_epochs': 1}

SINGLETASK_MULTIVALID_ARGS = {
    'task': 'integration_tests:nocandidate',
    'num_epochs': 2,
    'validation_every_n_epochs': 1,
    'validation_metric': 'ppl',
}

MULTITASK_ARGS = {
    'task': 'integration_tests:nocandidate,integration_tests:multiturn_nocandidate',
    'num_epochs': 1,
}

MULTITASK_MULTIVALID_ARGS = {
    'task': 'integration_tests:nocandidate,integration_tests:multiturn_nocandidate',
    'num_epochs': 2,
    'validation_every_n_epochs': 1,
    'validation_metric': 'ppl',
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
                args = BASE_ARGS.copy()
                args.update(extra_args)
                args['batchsize'] = bsz
                valid, test = testing_utils.train_model(args)
                for report in [valid, test]:
                    self.assertEqual(report['exs'], NUM_TEST, f'args: {args}')
                self.assertEqual(
                    valid['total_train_updates'], test['total_train_updates']
                )

    def test_singletask_dynb(self):
        """
        Test batchworld singletask.
        """
        for extra_args in [SINGLETASK_ARGS, SINGLETASK_MULTIVALID_ARGS]:
            for dyn_batch in ['full', 'batchsort', 'none']:
                args = BASE_ARGS.copy()
                args.update(extra_args)
                args['batchsize'] = 16
                args['dynamic_batching'] = dyn_batch
                valid, test = testing_utils.train_model(args)
                for report in [valid, test]:
                    self.assertEqual(report['exs'], NUM_TEST, f'args: {args}')
                self.assertEqual(
                    valid['total_train_updates'], test['total_train_updates']
                )

    def test_multitask(self):
        """
        Normal test - test if model can train via batchworld.
        """
        for extra_args in [MULTITASK_ARGS, MULTITASK_MULTIVALID_ARGS]:
            for bsz in [1, 16]:
                args = BASE_ARGS.copy()
                args.update(extra_args)
                args['batchsize'] = bsz
                valid, test = testing_utils.train_model(args)
                for rep in [valid, test]:
                    self.assertEqual(
                        rep['exs'], NUM_TEST + (NUM_TEST * 4), f'args: {args}'
                    )
                    self.assertEqual(
                        rep['tasks']['integration_tests:nocandidate']['exs'], NUM_TEST
                    )
                    self.assertEqual(
                        rep['tasks']['integration_tests:multiturn_nocandidate']['exs'],
                        NUM_TEST * 4,
                    )

                self.assertEqual(
                    valid['total_train_updates'], test['total_train_updates']
                )

    def test_multitask_dynamic_batching(self):
        """
        Multitask dynamic batching.
        """
        for extra_args in [MULTITASK_ARGS, MULTITASK_MULTIVALID_ARGS]:
            for dyn_batch in ['full', 'batchsort', 'none']:
                args = BASE_ARGS.copy()
                args.update(extra_args)
                args['batchsize'] = 16
                args['dynamic_batching'] = dyn_batch
                valid, test = testing_utils.train_model(args)
                for rep in [valid, test]:
                    self.assertEqual(
                        rep['exs'], NUM_TEST + (NUM_TEST * 4), f'args: {args}'
                    )
                    self.assertEqual(
                        rep['tasks']['integration_tests:nocandidate']['exs'], NUM_TEST
                    )
                    self.assertEqual(
                        rep['tasks']['integration_tests:multiturn_nocandidate']['exs'],
                        NUM_TEST * 4,
                    )

                self.assertEqual(
                    valid['total_train_updates'], test['total_train_updates']
                )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils

BASE_ARGS = {
    # Model Args
    'model': 'image_seq2seq',
    'embedding_size': 32,
    'n_heads': 2,
    'n_layers': 2,
    'n_positions': 128,
    'truncate': 128,
    'ffn_size': 128,
    'image_features_dim': 2048,
    'variant': 'xlm',
    'activation': 'gelu',
    'embeddings_scale': True,
    'gradient_clip': 0.1,
    'num_epochs': 10,
    # Train args
    'learningrate': 7e-3,
    'batchsize': 16,
    'optimizer': 'adamax',
    'learn_positional_embeddings': True,
}

TEXT_ARGS = {'task': 'integration_tests:nocandidate', 'num_epochs': 4}

IMAGE_ARGS = {
    'task': 'integration_tests:ImageTeacher',
    'num_epochs': 20,
    'image_mode': 'resnet152',
}

MULTITASK_ARGS = {
    'task': ','.join([m['task'] for m in [IMAGE_ARGS, TEXT_ARGS]]),  # type: ignore
    'num_epochs': 10,
    'multitask_weights': [1, 50],
    'image_mode': 'resnet152',
}

EVAL_ARGS = {
    'task': 'integration_tests:nocandidate',
    'skip_generation': False,
    'inference': 'beam',
    'beam_size': 2,
    'metrics': 'all',
    'compute_tokenized_bleu': True,
}


@testing_utils.skipUnlessTorch14
class TestImageSeq2Seq(unittest.TestCase):
    """
    Unit tests for the ImageSeq2Seq Agent.

    Mostly testing that the agent cooperates with tasks accordingly.
    """

    @testing_utils.retry(ntries=3)
    def test_text_task(self):
        """
        Test that model correctly handles text task.
        """
        args = BASE_ARGS.copy()
        args.update(TEXT_ARGS)
        valid, test = testing_utils.train_model(args)
        self.assertLessEqual(
            valid['ppl'], 1.5, 'failed to train image_seq2seq on text task'
        )

    @testing_utils.retry(ntries=3)
    @testing_utils.skipUnlessTorch
    @testing_utils.skipUnlessGPU
    def test_image_task(self):
        """
        Test that model correctly handles image task.

        No training, only eval
        """
        args = BASE_ARGS.copy()
        args.update(IMAGE_ARGS)

        valid, test = testing_utils.train_model(args)
        self.assertLessEqual(
            valid['ppl'], 8.6, 'failed to train image_seq2seq on image task'
        )

    @testing_utils.retry(ntries=3)
    @testing_utils.skipUnlessTorch
    @testing_utils.skipUnlessGPU
    def test_multitask(self):
        """
        Test that model can handle multiple inputs.
        """
        args = BASE_ARGS.copy()
        args.update(MULTITASK_ARGS)

        valid, test = testing_utils.train_model(args)
        self.assertLessEqual(
            valid['ppl'], 5.0, 'failed to train image_seq2seq on image+text task',
        )


if __name__ == '__main__':
    unittest.main()

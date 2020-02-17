#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils

"""
Integration tests for the Unlikelihood project
"""


FAST_MODE = True
NUM_EXAMPLES = 512 if FAST_MODE else -1


@testing_utils.skipUnlessGPU
class TestUnlikelihood(unittest.TestCase):
    def test_train_model_repeat_ul(self):
        """
        Check the training script doesn't crash.
        """
        from parlai.scripts.train_model import TrainLoop

        parser = ParlaiParser()
        # make it much smaller just for testing
        parser.set_params(
            model='projects.unlikelihood.agents:RepetitionUnlikelihoodParlallAgent',
            dict_file='izoo:unlikelihood/repeats/convai2/contextonly/model.dict'
            max_train_time=120,
            validation_max_exs=128,
            batchsize=16,
            truncate=32,
            short_final_eval=True,
        )
        opt = parser.parse_args([])
        TrainLoop(opt).train()

    def test_repeat_convai_contextonly(self):
        """
        Check the training script doesn't crash.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'izoo:unlikelihood/repeats/convai2/contextonly/model',
                'model': 'projects.unlikelihood.agents:RepetitionUnlikelihoodParlallAgent',
                'task': 'convai2',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
            },
            skip_test=True,
        )
        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 11.88, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2047, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 11.76, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1937, delta=0.0002)

    def test_repeat_eli5_contextonly(self):
        """
        Check the training script doesn't crash.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'izoo:unlikelihood/repeats/eli5/contextonly/model',
                'task': 'internal:eli5',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 21.71, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1629, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 21.37, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1628, delta=0.0002)

    def test_repeat_wiz_contextonly(self):
        """
        Check the training script doesn't crash.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'izoo:unlikelihood/repeats/wiz/contextonly/model',
                'task': 'wizard_of_wikipedia:GeneratorTeacher',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
                'prepend_gold_knowledge': True,
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 8.698, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3430, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 8.761, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3456, delta=0.0002)


if __name__ == '__main__':
    unittest.main()

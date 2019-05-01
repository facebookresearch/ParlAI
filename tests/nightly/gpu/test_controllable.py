#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.core.testing_utils as testing_utils

"""
Integration tests for the Controllable Dialogue project.

See projects/controllable_dialogue.
"""


@testing_utils.skipUnlessGPU
class TestControllableDialogue(unittest.TestCase):
    def test_dataset_integrity(self):
        """
        Check the controllble dialogue data loads.
        """
        train_output, valid_output, _ = testing_utils.display_data({
            'task': 'projects.controllable_dialogue.tasks.agents',
        })

        # check valid data
        self.assertIn('[lastuttsim]', train_output)
        self.assertIn(
            "hi , how are you doing ? i'm getting ready to do some cheetah "
            "chasing to stay in shape .",
            train_output
        )
        self.assertIn('131438 examples', train_output)

        # check valid data
        self.assertIn(
            "hello what are doing today ?",
            valid_output
        )
        self.assertIn('[lastuttsim]', valid_output)
        self.assertIn('7801 examples', valid_output)

    def test_train_model(self):
        """
        Check the training script doesn't crash.
        """
        import projects.controllable_dialogue.train_controllable_seq2seq as tcs2s
        parser = tcs2s.setup_args()
        # make it much smaller just for testing
        parser.set_params(
            max_train_time=120,
            validation_max_exs=128,
            batchsize=16,
            truncate=32,
            short_final_eval=True,
        )
        with testing_utils.capture_output():
            opt = parser.parse_args()
            tcs2s.TrainLoop(opt).train()

    def test_convai2_finetuned_greedy(self):
        """
        Check the greedy model produces correct results.
        """
        _, valid, _ = testing_utils.eval_model({
            'model_file': 'models:controllable_dialogue/convai2_finetuned_baseline',
            'task': 'projects.controllable_dialogue.tasks.agents',
            'beam_size': 1,
            'batchsize': 64,
        }, skip_test=True)
        self.assertAlmostEqual(
            valid['ppl'],
            22.86,
            delta=0.1,
        )
        self.assertAlmostEqual(
            valid['f1'],
            0.1702,
            delta=0.0002,
        )

    def test_convai2_finetuned_beamsearch(self):
        """
        Check the beamsearch baseline produces correct results.
        """
        _, valid, _ = testing_utils.eval_model({
            'model_file': 'models:controllable_dialogue/convai2_finetuned_baseline',
            'task': 'projects.controllable_dialogue.tasks.agents',
            'beam_size': 20,
            'beam_min_n_best': 10,
            'batchsize': 64,
            'num_examples': 512,  # don't run on the full dataset
        }, skip_test=True)
        self.assertAlmostEqual(
            valid['ppl'],
            23.54,  # 22.86 on the full dataset
            delta=0.1,
        )
        self.assertAlmostEqual(
            valid['f1'],
            0.1575,  # 0.1516 on the full dataset
            delta=0.0002,
        )

    def test_convai2_finetuned_norepetition(self):
        """
        Checks the finetuned model with repetition blocking produces correct results.
        """
        pass

    def test_ct_questionb11e10(self):
        """
        Checks the question-controlled model produces correct results.
        """
        pass

    def test_ct_avgnidf10b10e(self):
        """
        Checks the specificity-CT model produces correct results.
        """
        pass

    def test_bfw_specificity(self):
        """
        Checks the specificity-BFW model produces correct results.
        """
        pass

    def test_bfw_responsiveness(self):
        """
        Checks the responsiveness-BFW model produces correct results.
        """
        pass


if __name__ == '__main__':
    unittest.main()

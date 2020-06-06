#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import parlai.utils.testing as testing_utils

"""
Integration tests for the Controllable Dialogue project.

See projects/controllable_dialogue.
"""


FAST_MODE = True
NUM_EXAMPLES = 512 if FAST_MODE else -1
NO_REPETITION = 'extrep_2gram:-3.5,extrep_nonstopword:-1e20,intrep_nonstopword:-1e20'


@testing_utils.skipUnlessGPU
class TestControllableDialogue(unittest.TestCase):
    def test_dataset_integrity(self):
        """
        Check the controllble dialogue data loads.
        """
        train_output, valid_output, _ = testing_utils.display_data(
            {
                'task': 'projects.controllable_dialogue.tasks.agents',
                'display_verbose': True,
            }
        )

        # check valid data
        self.assertIn('[lastuttsim]', train_output)
        self.assertIn(
            "hi , how are you doing ? i'm getting ready to do some cheetah "
            "chasing to stay in shape .",
            train_output,
        )
        self.assertIn('131438 examples', train_output)

        # check valid data
        self.assertIn("hello what are doing today ?", valid_output)
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
            embedding_type='random',
        )
        opt = parser.parse_args([])
        tcs2s.TrainLoop(opt).train()

    def test_convai2_finetuned_greedy(self):
        """
        Check the greedy model produces correct results.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:controllable_dialogue/convai2_finetuned_baseline',
                'task': 'projects.controllable_dialogue.tasks.agents',
                'beam_size': 1,
                'batchsize': 64,
            },
            skip_test=True,
        )

        self.assertAlmostEqual(valid['ppl'], 22.86, delta=0.1)
        self.assertAlmostEqual(valid['f1'], 0.1702, delta=0.0002)

    def test_convai2_finetuned_beamsearch(self):
        """
        Check the beamsearch baseline produces correct results.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:controllable_dialogue/convai2_finetuned_baseline',
                'task': 'projects.controllable_dialogue.tasks.agents',
                'beam_size': 20,
                'beam_min_n_best': 10,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 23.54, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1575, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 22.86, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1516, delta=0.0002)

    def test_convai2_finetuned_norepetition(self):
        """
        Checks the finetuned model with repetition blocking produces correct results.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:controllable_dialogue/convai2_finetuned_baseline',
                'task': 'projects.controllable_dialogue.tasks.agents',
                'beam_size': 20,
                'beam_min_n_best': 10,
                'use_reply': 'model',
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
                'weighted_decoding': NO_REPETITION,
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 26.66, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1389, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 25.83, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1375, delta=0.0002)

    def test_ct_question_bucket7(self):
        """
        Checks the question-controlled model (z=7) produces correct results.
        """
        valid, _ = testing_utils.eval_model(
            {
                # b11e10 stands for 11 buckets, embedding size 10
                'model_file': 'zoo:controllable_dialogue/control_questionb11e10',
                'task': 'projects.controllable_dialogue.tasks.agents',
                'beam_size': 20,
                'beam_min_n_best': 10,
                'batchsize': 64,
                'use_reply': 'model',
                'num_examples': NUM_EXAMPLES,
                'weighted_decoding': NO_REPETITION,
                'set_controls': 'question:7',
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 31.04, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1362, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 29.22, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1336, delta=0.0002)

    def test_ct_question_bucket10(self):
        """
        Checks the question-controlled model (z=10 boost) produces correct results.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:controllable_dialogue/control_questionb11e10',
                'task': 'projects.controllable_dialogue.tasks.agents',
                'beam_size': 20,
                'beam_min_n_best': 10,
                'batchsize': 64,
                'use_reply': 'model',
                'num_examples': NUM_EXAMPLES,
                'weighted_decoding': 'extrep_nonstopword:-1e20,intrep_nonstopword:-1e20',
                'set_controls': 'question:10',
                'beam_reorder': 'best_extrep2gram_qn',
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 31.27, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1400, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 30.26, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1304, delta=0.0002)

    def test_ct_specificity_bucket7(self):
        """
        Checks the specificity-CT model (z=7) produces correct results.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:controllable_dialogue/control_avgnidf10b10e',
                'task': 'projects.controllable_dialogue.tasks.agents',
                'beam_size': 20,
                'beam_min_n_best': 10,
                'use_reply': 'model',
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
                'weighted_decoding': NO_REPETITION,
                'set_controls': 'avg_nidf:7',
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 38.64, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1376, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 37.03, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1365, delta=0.0002)

    def test_wd_specificity(self):
        """
        Checks the specificity-weighted decoding model produces correct results.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:controllable_dialogue/convai2_finetuned_baseline',
                'task': 'projects.controllable_dialogue.tasks.agents',
                'beam_size': 20,
                'beam_min_n_best': 10,
                'use_reply': 'model',
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
                'weighted_decoding': NO_REPETITION + ',nidf:4',
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 25.74, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1366, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 25.57, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1349, delta=0.0002)

    def test_wd_responsiveness(self):
        """
        Checks the responsiveness-weighted decoding model produces correct results.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:controllable_dialogue/convai2_finetuned_baseline',
                'task': 'projects.controllable_dialogue.tasks.agents',
                'beam_size': 20,
                'beam_min_n_best': 10,
                'use_reply': 'model',
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
                'weighted_decoding': NO_REPETITION
                + ',intrep_2gram:-1e20,partnerrep_2gram:-1e20,lastuttsim:5',  # noqa: E501
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 26.16, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1399, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 25.47, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1369, delta=0.0002)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import parlai.utils.testing as testing_utils

"""
Integration tests for the Dialogue Unlikelihood project
"""


FAST_MODE = True
NUM_EXAMPLES = 128 if FAST_MODE else -1


@testing_utils.skipUnlessGPU
class TestUnlikelihood(unittest.TestCase):
    def test_train_model_repeat_ul(self):
        """
        Check the training script doesn't crash.
        """
        opt = {
            'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
            'load_from_checkpoint': False,
            'task': 'convai2',
            'max_train_steps': 10,
            'validation_max_exs': 64,
            'batchsize': 16,
            'truncate': 32,
            'short_final_eval': True,
        }
        testing_utils.train_model(opt)

    def test_train_model_vocab_ul(self):
        """
        Check the training script doesn't crash.
        """
        with testing_utils.tempdir() as tmpdir:
            fp = os.path.join(tmpdir, "counts.txt")
            with open(fp, "w") as f:
                f.write(
                    '{"word": "test", "word_id": 0, "count": 1, "prob": 1, "cumprob": 1, "bin": "frequent"}'
                )
            opt = {
                'model': 'projects.dialogue_unlikelihood.agents:TransformerSequenceVocabUnlikelihoodAgent',
                'load_from_checkpoint': False,
                'task': 'convai2',
                'max_train_steps': 10,
                'validation_max_exs': 64,
                'batchsize': 16,
                'truncate': 32,
                'short_final_eval': True,
                'label_truncate': 256,
                'counts_file': fp,
            }
            testing_utils.train_model(opt)

    def test_repeat_convai_contextonly(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/rep_convai2_ctxt/model',
                'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
                'task': 'convai2',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
            },
            skip_test=True,
        )
        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 11.01, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2166, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 11.76, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1937, delta=0.0002)

    def test_repeat_convai_labelonly(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/rep_convai2_label/model',
                'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
                'task': 'convai2',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
            },
            skip_test=True,
        )
        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 10.50, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2332, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 11.42, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2029, delta=0.0002)

    def test_repeat_convai_contextandlabel(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/rep_convai2_ctxt_and_label/model',
                'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
                'task': 'convai2',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
            },
            skip_test=True,
        )
        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 10.95, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2220, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 11.85, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1925, delta=0.0002)

    def test_repeat_wiki_contextonly(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/rep_wiki_ctxt/model',
                'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
                'task': 'wizard_of_wikipedia:GeneratorTeacher',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
                'prepend_gold_knowledge': True,
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 8.071, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3472, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 8.761, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3456, delta=0.0002)

    def test_repeat_wiki_labelonly(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/rep_wiki_label/model',
                'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
                'task': 'wizard_of_wikipedia:GeneratorTeacher',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
                'prepend_gold_knowledge': True,
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 7.667, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3841, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 8.326, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3714, delta=0.0002)

    def test_repeat_wiki_contextandlabel(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/rep_wiki_ctxt_and_label/model',
                'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
                'task': 'wizard_of_wikipedia:GeneratorTeacher',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
                'prepend_gold_knowledge': True,
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 7.861, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3778, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 8.498, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3582, delta=0.0002)

    def test_vocab_alpha1e0(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/vocab_alpha1e0/model',
                'model': 'projects.dialogue_unlikelihood.agents:TransformerSequenceVocabUnlikelihoodAgent',
                'task': 'convai2',
                'beam_size': 1,
                'batchsize': 12,
                'num_examples': NUM_EXAMPLES,
                'skip_generation': False,
            },
            skip_test=True,
        )
        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 9.664, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2165, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 11.42, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2004, delta=0.0002)

    def test_vocab_alpha1e1(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/vocab_alpha1e1/model',
                'model': 'projects.dialogue_unlikelihood.agents:TransformerSequenceVocabUnlikelihoodAgent',
                'task': 'convai2',
                'beam_size': 1,
                'batchsize': 12,
                'num_examples': NUM_EXAMPLES,
                'skip_generation': False,
            },
            skip_test=True,
        )
        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 10.05, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2337, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 11.82, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2009, delta=0.0002)


if __name__ == '__main__':
    unittest.main()

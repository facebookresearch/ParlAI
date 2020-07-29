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
NUM_EXAMPLES = 512 if FAST_MODE else -1


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
            'max_train_time': 120,
            'validation_max_exs': 128,
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
                'max_train_time': 120,
                'validation_max_exs': 128,
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
            self.assertAlmostEqual(valid['ppl'], 11.88, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2047, delta=0.0002)
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
            self.assertAlmostEqual(valid['ppl'], 11.46, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2130, delta=0.0002)
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
            self.assertAlmostEqual(valid['ppl'], 11.98, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2034, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 11.85, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1925, delta=0.0002)

    @unittest.skip
    def test_repeat_eli5_contextonly(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/rep_eli5_ctxt/model',
                'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
                'task': 'eli5',
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

    @unittest.skip
    def test_repeat_eli5_labelonly(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/rep_eli5_label/model',
                'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
                'task': 'eli5',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 21.71, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1777, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 21.39, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1825, delta=0.0002)

    @unittest.skip
    def test_repeat_eli5_contextandlabel(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/rep_eli5_ctxt_and_label/model',
                'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent',
                'task': 'eli5',
                'beam_size': 1,
                'batchsize': 64,
                'num_examples': NUM_EXAMPLES,
            },
            skip_test=True,
        )

        if FAST_MODE:
            self.assertAlmostEqual(valid['ppl'], 22.13, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1805, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 21.80, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1843, delta=0.0002)

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
            self.assertAlmostEqual(valid['ppl'], 8.698, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3430, delta=0.0002)
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
            self.assertAlmostEqual(valid['ppl'], 8.284, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3744, delta=0.0002)
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
            self.assertAlmostEqual(valid['ppl'], 8.433, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.3614, delta=0.0002)
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
            self.assertAlmostEqual(valid['ppl'], 11.26, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2115, delta=0.0002)
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
            self.assertAlmostEqual(valid['ppl'], 11.66, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2118, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 11.82, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.2009, delta=0.0002)

    def test_vocab_alpha1e2(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/vocab_alpha1e2/model',
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
            self.assertAlmostEqual(valid['ppl'], 12.38, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1997, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 12.48, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1903, delta=0.0002)

    def test_vocab_alpha1e3(self):
        """
        Verify recorded ppl and F1 scores for released models.
        """
        valid, _ = testing_utils.eval_model(
            {
                'model_file': 'zoo:dialogue_unlikelihood/vocab_alpha1e3/model',
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
            self.assertAlmostEqual(valid['ppl'], 14.12, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1872, delta=0.0002)
        else:
            self.assertAlmostEqual(valid['ppl'], 14.27, delta=0.1)
            self.assertAlmostEqual(valid['f1'], 0.1734, delta=0.0002)


if __name__ == '__main__':
    unittest.main()

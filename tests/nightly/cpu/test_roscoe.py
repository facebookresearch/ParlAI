#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from unittest.mock import patch

import numpy as np

try:
    import transformers  # noqa
    from sentence_transformers import SentenceTransformer  # noqa
    from simcse import SimCSE  # noqa
    from projects.roscoe.score import (
        Chain,
        Evaluator,
        LANGUAGE_MODEL_SCORES,
        PPL_CHAIN,
        PPL_STEP,
        PPL_STEP_MAX,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE, 'Must install transformers and simcse to run this test'
)
class TestEvaluator(unittest.TestCase):
    """
    Basic tests for some simple functionality.
    """

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.reasoning_evaluator = Evaluator(
            hypos=[],
            context=[],
            references=[],
            score_types=[],
        )
        self.reasoning_evaluator.ppl_batch = 1
        self.chain = Chain("chain")

    def test_linearize_array(self):
        pre_linearized = [['1', '2'], ['3', '4']]
        desired_linearized = ['1', '2', '3', '4']
        linearized = self.reasoning_evaluator.linearize_array(pre_linearized)
        self.assertEqual(desired_linearized, linearized)

    def test_repetitions_words(self):
        input_word_embeddings = np.array(
            [
                np.array([[0.1, 0.1], [0.2, 0.2]]),  # sentence 1
                np.array([[0.1, 0.1]]),  # sentence 2
            ]
        )
        desired_similarity = 1.0
        similarity = self.reasoning_evaluator.repetitions(input_word_embeddings)
        self.assertAlmostEqual(desired_similarity, similarity, places=2)

        input_word_embeddings = np.array(
            [
                np.array([[0.1, 0.1]]),  # sentence 1
                np.array([[-0.1, -0.1]]),  # sentence 2
            ]
        )
        desired_similarity = 0.0
        similarity = self.reasoning_evaluator.repetitions(input_word_embeddings)
        self.assertAlmostEqual(desired_similarity, similarity, places=2)

    def test_repetitions_sentences(self):
        input_sentence_embeddings = np.array(
            [
                np.array([0.1, 0.1]),  # sentence 1
                np.array([0.1, 0.1]),  # sentence 2
                np.array([0.2, 0.2]),  # sentence 3
            ]
        )
        desired_similarity = 1.0
        similarity = self.reasoning_evaluator.repetitions(input_sentence_embeddings)
        self.assertAlmostEqual(desired_similarity, similarity, places=2)

        input_sentence_embeddings = np.array(
            [np.array([0.1, 0.1]), np.array([-0.1, -0.1])]  # sentence 1  # sentence 2
        )
        desired_similarity = 0.0
        similarity = self.reasoning_evaluator.repetitions(input_sentence_embeddings)
        self.assertAlmostEqual(desired_similarity, similarity, places=2)

    @patch("projects.roscoe.score.Evaluator.contradiction_probability")
    def test_max_contradiction(self, mock_contradiction_probability):
        mock_contradiction_probability.return_value = [1.0]
        context = ["a", "b"]
        hypo = ["c", "d", "e"]
        batch_size = 1
        desired_probs_size = len(context) * len(hypo)
        probs = self.reasoning_evaluator.max_contradiction(context, hypo, batch_size)
        self.assertEqual(desired_probs_size, len(probs))

    @patch("projects.roscoe.score.Evaluator.perplexity")
    def test_compute_ppl_scores(self, mock_perplexity):
        mock_perplexity.return_value = [(2, 2), (20, 2)]
        score_types = LANGUAGE_MODEL_SCORES
        desired_scores = {
            PPL_CHAIN: 0.5,
            PPL_STEP: 0.09,
            PPL_STEP_MAX: 0.05,
        }
        scores = self.reasoning_evaluator.compute_ppl_scores(self.chain, score_types)
        for model in LANGUAGE_MODEL_SCORES:
            self.assertTrue(model in scores)
            self.assertAlmostEqual(desired_scores[model], scores[model], places=2)


if __name__ == '__main__':
    unittest.main()

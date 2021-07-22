#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test agents used with style-controlled generation.
"""

import unittest

import parlai.utils.testing as testing_utils
from parlai.core.opt import Opt


class TestClassifierOnGenerator(unittest.TestCase):
    """
    Test classifier on generator.
    """

    @testing_utils.retry()
    def test_simple(self):
        valid, test = testing_utils.train_model(
            Opt(
                dict(
                    task='integration_tests:classifier',
                    model='projects.style_gen.classifier:ClassifierAgent',
                    classes=['one', 'zero'],
                    optimizer='adamax',
                    truncate=8,
                    learningrate=7e-3,
                    batchsize=16,
                    num_epochs=5,
                    n_layers=1,
                    n_heads=1,
                    ffn_size=32,
                    embedding_size=32,
                )
            )
        )
        self.assertEqual(valid['accuracy'], 1.0)
        self.assertEqual(test['accuracy'], 1.0)

    def test_accuracy(self):
        """
        Test the accuracy of the classifier trained on previous and current utterances.

        This should be very close to 100%, because this classifier was used to label the
        styles of this dataset to begin with.
        """
        _, test = testing_utils.eval_model(
            opt={
                'batchsize': 4,
                'fp16': True,
                'num_examples': 16,
                'model_file': 'zoo:style_gen/prev_curr_classifier/model',
                'model': 'projects.style_gen.classifier:ClassifierAgent',
                'classes_from_file': 'image_chat_personalities_file',
                'task': 'style_gen:PrevCurrUttStyle',
                'wrapper_task': 'style_gen:LabeledBlendedSkillTalk',
            },
            skip_valid=True,
        )
        self.assertAlmostEqual(test['accuracy'], 1.0, delta=0.0)


class TestStyleGen(unittest.TestCase):
    def test_perplexities(self):
        """
        Test perplexities of style-controlled generation models in the zoo.
        """
        test_cases = [('c75_labeled_dialogue_generator', 1.0, 7.664)]
        for model_name, style_frac, desired_ppl in test_cases:
            _, test = testing_utils.eval_model(
                opt={
                    'batchsize': 4,
                    'fp16': True,
                    'num_examples': 16,
                    'model_file': f'zoo:style_gen/{model_name}/model',
                    'model': 'projects.style_gen.style_gen:StyleGenAgent',
                    'skip_generation': True,
                    'task': 'style_gen:LabeledBlendedSkillTalk',
                    'use_style_frac': style_frac,
                },
                skip_valid=True,
            )
            self.assertAlmostEqual(test['ppl'], desired_ppl, delta=0.005)


if __name__ == '__main__':
    unittest.main()

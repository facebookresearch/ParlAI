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


class TestStyleGen(unittest.TestCase):
    def test_perplexities(self):
        """
        Test perplexities of style-controlled generation models in the zoo.
        """
        test_cases = [('c75_labeled_dialogue_generator', 1.0, 9.442)]
        for model_name, style_frac, desired_ppl in test_cases:
            _, test = testing_utils.eval_model(
                opt={
                    'batchsize': 16,
                    'num_examples': 64,
                    'model_file': f'zoo:style_gen/{model_name}/model',
                    'model': 'style_gen',
                    'skip_generation': True,
                    'task': 'style_gen:LabeledBlendedSkillTalk',
                    'use_style_frac': style_frac,
                },
                skip_valid=True,
            )
            self.assertAlmostEqual(test['ppl'], desired_ppl, delta=0.005)


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
                    model='style_gen/classifier',
                    classes=['one', 'zero'],
                    optimizer='adamax',
                    truncate=8,
                    learningrate=7e-3,
                    batchsize=32,
                    num_epochs=5,
                    n_layers=1,
                    n_heads=1,
                    ffn_size=32,
                    embedding_size=32,
                )
            )
        )
        assert valid['accuracy'] > 0.97
        assert test['accuracy'] > 0.97


if __name__ == '__main__':
    unittest.main()

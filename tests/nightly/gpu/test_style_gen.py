#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test agents used with style-controlled generation.
"""

import unittest

import parlai.utils.testing as testing_utils


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
                    'fp16': False,
                    'num_examples': 16,
                    'model_file': f'zoo:style_gen/{model_name}/model',
                    'model': 'projects.style_gen.style_gen:StyleGenAgent',
                    'skip_generation': True,
                    'task': 'style_gen:LabeledBlendedSkillTalk',
                    'use_style_frac': style_frac,
                },
                skip_valid=True,
            )
            # We turn off FP16 because emulation of this is likely slow on the CI GPUs
            self.assertAlmostEqual(test['ppl'], desired_ppl, delta=0.005)


if __name__ == '__main__':
    unittest.main()

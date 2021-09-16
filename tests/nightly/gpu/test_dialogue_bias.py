#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test models with reduced bias.
"""

import unittest

import parlai.utils.testing as testing_utils
from parlai.core.opt import Opt


class TestDialogueBias(unittest.TestCase):
    def test_perplexities(self):
        """
        Test perplexities of reduced-bias models in the zoo.
        """
        test_cases = [
            ('gender__name_scrambling', 'transformer/generator', 22.91),
            (
                'gender__ctrl_gen_tokens',
                'projects.dialogue_bias.agents:NoBiasStyleGenAgent',
                22.61,
            ),
            ('gender__unlikelihood_sequence_level', 'transformer/generator', 11.44),
            ('gender_ethnicity__name_scrambling', 'transformer/generator', 19.57),
        ]
        # Perplexities are high because models were tuned on conversations starting with
        # "Hi! My name is ___."
        for model_name, model, desired_ppl in test_cases:
            _, test = testing_utils.eval_model(
                opt={
                    'batchsize': 4,
                    'beam_block_full_context': True,
                    'fp16': True,
                    'num_examples': 16,
                    'model': model,
                    'model_file': f'zoo:dialogue_bias/{model_name}/model',
                    'skip_generation': True,
                    'task': 'blended_skill_talk',
                },
                skip_valid=True,
            )
            self.assertAlmostEqual(test['ppl'], desired_ppl, delta=0.05)


if __name__ == '__main__':
    unittest.main()

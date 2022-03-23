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
    def _test_perplexity(self, model_name, model, desired_ppl):
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

    def test_gender__name_scrambling(self):
        self._test_perplexity('gender__name_scrambling', 'transformer/generator', 22.91)

    def test_gender__ctrl_gen_tokens(self):
        self._test_perplexity(
            'gender__ctrl_gen_tokens',
            'projects.dialogue_bias.agents:NoBiasStyleGenAgent',
            22.61,
        )

    def test_gender__unlikelihood_sequence_leve(self):
        self._test_perplexity(
            'gender__unlikelihood_sequence_level', 'transformer/generator', 11.44
        )

    def test_gender__ethnicity__name_scrambling(self):
        self._test_perplexity(
            'gender_ethnicity__name_scrambling', 'transformer/generator', 19.57
        )


if __name__ == '__main__':
    unittest.main()

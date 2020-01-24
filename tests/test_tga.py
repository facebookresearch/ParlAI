#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test TorchGeneratorAgent.
"""

import unittest
from parlai.core.agents import create_agent
import parlai.utils.testing as testing_utils
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import TorchGeneratorAgent


class TestUpgradeOpt(unittest.TestCase):
    """
    Test upgrade_opt behavior.
    """

    def test_inference(self):
        """
        Test --inference with simple options.
        """
        upgraded = TorchGeneratorAgent.upgrade_opt({'beam_size': 1})
        self.assertEqual(upgraded['inference'], 'greedy')

        upgraded = TorchGeneratorAgent.upgrade_opt({'beam_size': 5})
        self.assertEqual(upgraded['inference'], 'beam')

    def test_no_greedy_largebeam(self):
        """
        Ensures that --beam-size > 1 and --inference greedy causes a failure.
        """
        # we should have an exception if we mix beam size > 1 with inference greedy
        with self.assertRaises(ValueError):
            testing_utils.display_model(
                dict(
                    task='integration_tests:multiturn_nocandidate',
                    model_file='zoo:unittest/transformer_generator2/model',
                    beam_size=5,
                    inference='greedy',
                )
            )

        # and we shouldn't if we have inference beam
        testing_utils.display_model(
            dict(
                task='integration_tests:multiturn_nocandidate',
                model_file='zoo:unittest/transformer_generator2/model',
                beam_size=5,
                inference='beam',
            )
        )

    def test_file_inference(self):
        """
        Test --inference with older model files.
        """
        pp = ParlaiParser(True, True)
        opt = pp.parse_args(
            ['--model-file', 'zoo:unittest/transformer_generator2/model']
        )
        agent = create_agent(opt, True)
        self.assertEqual(agent.opt['inference'], 'greedy')

        pp = ParlaiParser(True, True)
        opt = pp.parse_args(
            [
                '--model-file',
                'zoo:unittest/transformer_generator2/model',
                '--beam-size',
                '5',
            ],
            print_args=False,
        )
        agent = create_agent(opt, True)
        self.assertEqual(agent.opt['inference'], 'beam')


if __name__ == '__main__':
    unittest.main()

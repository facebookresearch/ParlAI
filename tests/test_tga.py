#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test TorchGeneratorAgent."""

import unittest
from parlai.core.agents import create_agent
import parlai.core.testing_utils as testing_utils
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import TorchGeneratorAgent


class TestUpgradeOpt(unittest.TestCase):
    """Test upgrade_opt behavior."""

    def test_inference(self):
        """Test --inference with simple options."""
        with testing_utils.capture_output():
            upgraded = TorchGeneratorAgent.upgrade_opt({'beam_size': 1})
            self.assertEqual(upgraded['inference'], 'greedy')

            upgraded = TorchGeneratorAgent.upgrade_opt({'beam_size': 5})
            self.assertEqual(upgraded['inference'], 'beam')

    def test_file_inference(self):
        """Test --inference with older model files."""
        testing_utils.download_unittest_models()
        with testing_utils.capture_output():
            pp = ParlaiParser(True, True)
            opt = pp.parse_args(
                ['--model-file', 'zoo:unittest/transformer_generator2/model']
            )
            agent = create_agent(opt, True)
            self.assertEqual(agent.opt['inference'], 'greedy')

        with testing_utils.capture_output():
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

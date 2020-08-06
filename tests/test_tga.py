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
            ]
        )
        agent = create_agent(opt, True)
        self.assertEqual(agent.opt['inference'], 'beam')

    def test_block_full_context(self):
        """
        Test --beam-block-full-context with older model files.
        """
        # old model file == beam block full context false
        pp = ParlaiParser(True, True)
        opt = pp.parse_args(
            ['--model-file', 'zoo:unittest/transformer_generator2/model']
        )
        agent = create_agent(opt, True)
        self.assertEqual(agent.opt['beam_block_full_context'], False)
        self.assertEqual(agent.beam_block_full_context, False)

        # brand new model == beam block full context true
        pp = ParlaiParser(True, True)
        opt = pp.parse_args(['--model', 'transformer/generator'])
        agent = create_agent(opt, True)
        self.assertEqual(agent.opt['beam_block_full_context'], True)
        self.assertEqual(agent.beam_block_full_context, True)


class TestTreeSearch(unittest.TestCase):
    """
    Tests various Tree Search functionalities.

    NOTE: Currently incomplete.
    """

    def test_full_context_block(self):
        args = [
            '--model-file',
            'zoo:unittest/transformer_generator2/model',
            '--inference',
            'beam',
            '--truncate',
            '1024',
        ]
        pp = ParlaiParser(True, True)
        agent = create_agent(pp.parse_args(args), True)
        obs = {'text': '1 2 3 4 ' * 256, 'episode_done': False}
        agent.observe(obs)
        batch = agent.batchify([agent.observation])
        self.assertEqual(agent._get_context(batch, 0).tolist(), [5, 4, 6, 7] * 256)

        # observe 1 more obs, context is the same (truncation)
        agent.observe(obs)
        batch = agent.batchify([agent.observation])
        self.assertEqual(agent._get_context(batch, 0).tolist(), [5, 4, 6, 7] * 256)

        # Now, set agent's beam_block_full_context
        args += ['--beam-block-full-context', 'true']
        agent2 = create_agent(pp.parse_args(args), True)
        agent2.observe(obs)
        batch = agent2.batchify([agent2.observation])
        self.assertEqual(agent2._get_context(batch, 0).tolist(), [5, 4, 6, 7] * 256)

        # observe 1 more obs, context is larger now
        agent2.observe(obs)
        batch = agent2.batchify([agent2.observation])
        self.assertEqual(
            agent2._get_context(batch, 0).tolist(),
            [5, 4, 6, 7] * 256 + [3] + [5, 4, 6, 7] * 256,
        )  # 3 is end token.


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test TorchGeneratorAgent.
"""
import unittest
import math
import torch
from parlai.core.agents import create_agent
import parlai.utils.testing as testing_utils
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import (
    BeamSearch,
    GreedySearch,
    NucleusSampling,
    TopKSampling,
    TorchGeneratorAgent,
)
from parlai.agents.test_agents.transformer_generator_prefix import PREFIX_TEXT


class TestTGA(unittest.TestCase):
    """
    Test various Torch Generator agent behaviors.
    """

    def test_upgrade_opt_inference(self):
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


class TestGeneration(unittest.TestCase):
    """
    Tests various generation functionalities.

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
            '--beam-context-block-ngram',
            '1',
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

    def test_prefix_tokens(self):
        """
        Test functionality of `get_prefix_tokens`.
        """
        args = [
            '--model-file',
            'zoo:unittest/transformer_generator2/model',
            '--model',
            'test_agents/transformer_generator_prefix',
            '--inference',
            'beam',
            '--truncate',
            '1024',
            '--beam-size',
            '2',
        ]
        pp = ParlaiParser(True, True)
        agent = create_agent(pp.parse_args(args), True)
        obs = {'text': '1 2 3 4 ' * 256, 'episode_done': False}
        agent.observe(obs)
        act = agent.act()
        beam_texts = [x[0] for x in act['beam_texts']]
        for beam in beam_texts:
            # check that all beams start with the prefix text
            assert beam.startswith(
                PREFIX_TEXT
            ), f"[{beam}] does not start with [{PREFIX_TEXT}]"

    def test_token_level_loss_logging(self):
        """
        Test functionality of token level probability + ranking logging.

        Regression for all inference types: 'beam', 'greedy', 'topk', 'nucleus',
        'delayedbeam'
        """
        inference_types = ['beam', 'greedy', 'topk', 'nucleus', 'delayedbeam']
        gold_data = {
            'beam': {
                'text_token_info': [
                    ('__start__', 0.0, 1.0),
                    ('5', -2.5510462364763953e-05, 0.0),
                    ('__end__', -1.1920922133867862e-06, 0.0),
                ],
                'extra_args': ['--beam-size', '3'],
            },
            'greedy': {
                'text_token_info': [
                    ('__start__', 0.0, 1.0),
                    ('5', -2.5510462364763953e-05, 0.0),
                    ('__end__', -1.1920922133867862e-06, 0.0),
                ],
                'extra_args': [],
            },
            # sampling based token selection will produce non-deterministic output, so we can't do data regression
            'topk': {'extra_args': ['--topk', '2']},
            'topk_multiple_beams': {'extra_args': ['--topk', '2', '--beam-size', '5']},
            # sampling based token selection will produce non-deterministic output, so we can't do data regression
            'nucleus': {'extra_args': ['--topp', '0.3']},
            'nucleus_multiple_beams': {
                'extra_args': ['--topp', '0.3', '--beam-size', '5']
            },
            # sampling based token selection will produce non-deterministic output, so we can't do data regression
            'delayedbeam': {'extra_args': ['--topk', '2', '--beam-delay', '2']},
        }

        for inference_type in inference_types:
            args = [
                '--model-file',
                'zoo:unittest/transformer_generator2/model',
                '--inference',
                inference_type,
                '--truncate',
                '1024',
                '-v',
            ] + gold_data[inference_type]['extra_args']

            pp = ParlaiParser(True, True)
            agent = create_agent(pp.parse_args(args), True)
            obs = {'text': '5', 'episode_done': False}
            agent.observe(obs)
            act = agent.act()

            if 'text_token_info' in gold_data[inference_type]:
                for i, tok_data in enumerate(act['text_token_info']):
                    assert (
                        gold_data[inference_type]['text_token_info'][i][0]
                        == tok_data[0]
                    ), f"failed token prediction for inference type {inference_type} at token {gold_data[inference_type]['text_token_info'][i][0]}"
                    assert math.isclose(
                        gold_data[inference_type]['text_token_info'][i][1], tok_data[1]
                    ), f"failed token probability prediction for inference type {inference_type} at token {gold_data[inference_type]['text_token_info'][i][0]}"
                    assert math.isclose(
                        gold_data[inference_type]['text_token_info'][i][2], tok_data[2]
                    ), f"failed token rank prediction for inference type {inference_type} at token {gold_data[inference_type]['text_token_info'][i][0]}"

    def test_tree_search(self):
        """
        Unit test `select_paths` for different decoding schemes.
        """
        tests = {
            "greedy": {
                "obj": GreedySearch(beam_size=1, verbose=True),
                "logprobs": torch.Tensor([[-1.0, -1.0, -0.1, -0.3]]),
                "prior_scores": torch.Tensor([-0.5]),
                "expected_result": {
                    "hypothesis_ids": torch.LongTensor([0]),
                    "token_ids": torch.LongTensor([2]),
                    "scores": torch.Tensor([-0.6]),
                    "token_scores": torch.Tensor([-0.1]),
                    "token_ranks": torch.LongTensor([0]),
                },
            },
            "beam_with_one_beam": {
                "obj": BeamSearch(beam_size=1, verbose=True),
                "logprobs": torch.Tensor([[-1.0, -1.0, -0.1, -0.3]]),
                "prior_scores": torch.Tensor([-0.5]),
                "expected_result": {
                    "hypothesis_ids": torch.LongTensor([0]),
                    "token_ids": torch.LongTensor([2]),
                    "scores": torch.Tensor([-0.6]),
                    "token_scores": torch.Tensor([-0.1]),
                    "token_ranks": torch.LongTensor([0]),
                },
            },
            "beam_with_multiple_beams": {
                "obj": BeamSearch(beam_size=2, verbose=True),
                "logprobs": torch.Tensor(
                    [[-0.1, -2.0, -3.0, -3.0], [-1.0, -1.0, -0.2, -0.3]]
                ),
                "prior_scores": torch.Tensor([-1.0, -0.5]),
                # logprobs + prior_scores = [[-1.1,-3.,-4.,-4.],[-1.5,-1.5,-0.7,-0.8]]
                "expected_result": {
                    "hypothesis_ids": torch.LongTensor([1, 1]),
                    "token_ids": torch.LongTensor([2, 3]),
                    "scores": torch.Tensor([-0.7, -0.8]),
                    "token_scores": torch.Tensor([-0.2, -0.3]),
                    "token_ranks": torch.LongTensor([0, 1]),
                },
            },
            "topk_with_one_beam": {
                "obj": TopKSampling(beam_size=1, k=3, verbose=True),
                "logprobs": torch.Tensor(
                    [[-float('inf'), -0.5, -float('inf'), -float('inf')]]
                ),
                "prior_scores": torch.Tensor([-3.0]),
                "expected_result": {
                    "hypothesis_ids": torch.LongTensor([0]),
                    "token_ids": torch.LongTensor([1]),
                    "scores": torch.Tensor([-3.5]),
                    "token_scores": torch.Tensor([-0.5]),
                    "token_ranks": torch.LongTensor([0]),
                },
            },
            "topk_with_multiple_beams": {
                "obj": TopKSampling(beam_size=2, k=3, verbose=True),
                "logprobs": torch.Tensor(
                    [
                        [-float('inf'), -0.5, -float('inf'), -float('inf')],
                        [-float('inf'), -float('inf'), -0.6, -float('inf')],
                    ]
                ),
                "prior_scores": torch.Tensor([-3.0, -2.0]),
                "expected_result": {
                    "hypothesis_ids": torch.LongTensor([0, 1]),
                    "token_ids": torch.LongTensor([1, 2]),
                    "scores": torch.Tensor([-3.5, -2.6]),
                    "token_scores": torch.Tensor([-0.5, -0.6]),
                    "token_ranks": torch.LongTensor([0, 0]),
                },
            },
            "nucleus_with_one_beam": {
                "obj": NucleusSampling(beam_size=1, p=0.9, verbose=True),
                "logprobs": torch.Tensor(
                    [[-float('inf'), -0.5, -float('inf'), -float('inf')]]
                ),
                "prior_scores": torch.Tensor([-3.0]),
                "expected_result": {
                    "hypothesis_ids": torch.LongTensor([0]),
                    "token_ids": torch.LongTensor([1]),
                    "scores": torch.Tensor(
                        [-3.0]
                    ),  # the -0.5 logprob normalizes to 0 in truncated distribution
                    "token_scores": torch.Tensor([-0.0]),  # same as above
                    "token_ranks": torch.LongTensor([0]),
                },
            },
            "nucleus_with_multiple_beams": {
                "obj": NucleusSampling(beam_size=2, p=0.9, verbose=True),
                "logprobs": torch.Tensor(
                    [
                        [-float('inf'), -0.5, -float('inf'), -float('inf')],
                        [-float('inf'), -float('inf'), -0.6, -float('inf')],
                    ]
                ),
                "prior_scores": torch.Tensor([-3.0, -2.0]),
                "expected_result": {
                    "hypothesis_ids": torch.LongTensor([0, 1]),
                    "token_ids": torch.LongTensor([1, 2]),
                    "scores": torch.Tensor(
                        [-3.0, -2.0]
                    ),  # the -0.5, -0.6 logprobs normalize to 0 in truncated distributions
                    "token_scores": torch.Tensor([-0.0, -0.0]),  # same as above
                    "token_ranks": torch.LongTensor([0, 0]),
                },
            },
        }

        for test_name, test_data in tests.items():
            path_selection = test_data["obj"].select_paths(
                test_data["logprobs"], test_data["prior_scores"], None
            )
            expected_result = test_data["expected_result"]

            assert torch.equal(
                path_selection.hypothesis_ids, expected_result["hypothesis_ids"]
            ), f"failed test_tree_search for test {test_name} on field hypothesis_ids"
            assert torch.equal(
                path_selection.token_ids, expected_result["token_ids"]
            ), f"failed test_tree_search for test {test_name} on field token_ids"
            assert torch.allclose(
                path_selection.scores, expected_result["scores"]
            ), f"failed test_tree_search for test {test_name} on field scores"
            assert torch.allclose(
                path_selection.token_scores, expected_result["token_scores"]
            ), f"failed test_tree_search for test {test_name} on field token_scores"
            assert torch.equal(
                path_selection.token_ranks, expected_result["token_ranks"]
            ), f"failed test_tree_search for test {test_name} on field token_ranks"


if __name__ == '__main__':
    unittest.main()

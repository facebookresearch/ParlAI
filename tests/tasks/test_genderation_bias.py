#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest

from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.tasks.genderation_bias.agents import ControllableTaskTeacher
from parlai.tasks.genderation_bias.utils import flatten_and_classify


class TestGenderationBiasTeacher(unittest.TestCase):
    """
    Tests for the Genderation Bias Teacher.

    For now, just test the flatten_and_classify utility function.
    """

    def test_flatten_and_classify(self):
        word_lists = ControllableTaskTeacher.build_wordlists(
            ParlaiParser().parse_args([])
        )
        utterances = [
            "hello there",
            "hi there dad, what's up",
            "not much, do you know where your sister is?",
            "I have not seen her, I thought she was with grandpa",
            "well, if you see her, let me know",
            "will do!",
            "ok, have a good day",
            "bye bye! tell mom I say hello",
        ]
        tokens = ['f0m1', 'f1m1', 'f0m0', 'f1m0']
        episode = [
            Message(
                {
                    'text': utterances[i],
                    'labels': [utterances[i + 1]],
                    'episode_done': False,
                }
            )
            for i in range(0, len(utterances) - 1, 2)
        ]
        episode[-1].force_set('episode_done', True)
        new_episode = flatten_and_classify(episode, -1, word_lists)
        assert len(new_episode) == 4
        assert all(
            ex['text'].endswith(tok) for ex, tok in zip(new_episode, tokens)
        ), f"new episode: {new_episode}"

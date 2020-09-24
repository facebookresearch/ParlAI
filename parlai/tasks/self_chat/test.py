#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.scripts.display_data import setup_args
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld

import unittest
from unittest.mock import MagicMock


class TestSelfChat(unittest.TestCase):
    def setUp(self):
        parser = setup_args()
        parser.set_defaults(
            interactive_mode=True,
            task='self_chat',
            selfchat_task=True,
            selfchat_max_turns=1,
        )
        self.opt = parser.parse_args([])

        agent1 = RepeatLabelAgent(self.opt)
        agent2 = agent1.clone()

        self.world = create_task(self.opt, [agent1, agent2])
        self.assertIsInstance(self.world, SelfChatBaseWorld)

    def test_seed_utterances(self):
        def assert_seed_utts_match(seed_utts):
            self.world.get_contexts = MagicMock()
            self.world.get_contexts.return_value = []
            self.world.get_openers = MagicMock()
            self.world.get_openers.return_value = seed_utts

            # Run a self chat
            self.world.parley()
            acts = self.world.get_acts()
            utterances = [a['text'] for a in acts]
            self.assertListEqual(seed_utts, utterances[: len(seed_utts)])

        assert_seed_utts_match(['hey'])
        assert_seed_utts_match(['hey', 'hi'])

    def test_contexts(self):
        def assert_contexts_match(contexts):
            self.world.get_contexts = MagicMock()
            self.world.get_contexts.return_value = contexts
            self.world.get_openers = MagicMock()
            self.world.get_openers.return_value = []

            # Run a self chat
            self.world.parley()
            acts = self.world.get_acts()
            utterances = [a['text'] for a in acts]
            self.assertSetEqual(set(contexts), set(utterances[: len(contexts)]))

        assert_contexts_match(['you are a seal', 'you are an ostrich'])
        assert_contexts_match([])


if __name__ == '__main__':
    unittest.main()

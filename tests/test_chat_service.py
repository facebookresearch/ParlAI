#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from parlai.chat_service.utils.config import WorldConfig
from parlai.chat_service.services.messenger.agents import MessengerAgent
from parlai.chat_service.services.messenger.messenger_manager import MessengerManager
from parlai.core.params import ParlaiParser

AGENT_ID = -1
TASK_ID = 'test_task_1'
PAGE_ID = 0
# config taken from `chat_service/tasks/overworld_demo`
OPT = {
    'is_debug': True,
    'config': {
        'overworld': 'MessengerOverworld',
        'world_path': 'parlai.chat_service.tasks.overworld_demo.worlds',
        'max_workers': 2,
        'task_name': 'overworld_demo',
        'configs': {
            'echo': WorldConfig(
                world_name='echo',
                onboarding_name='MessengerEchoOnboardWorld',
                task_name='MessengerEchoTaskWorld',
                max_time_in_pool=180,
                agents_required=1,
                backup_task=None,
            )
        },
        'additional_args': {'page_id': PAGE_ID},
    },
    'bypass_server_setup': True,
}


class TestChatServiceAgent(unittest.TestCase):
    """
    Tests for chat service agent functionality.
    """

    def _get_args(self):
        parser = ParlaiParser(False, False)
        parser.add_parlai_data_path()
        parser.add_messenger_args()
        return parser.parse_args([])

    def test_data_persistence(self):
        """
        Test to make sure that assigning the `data` attribute of a MessengerAgent does
        not overwrite values already in `data`.
        """
        opt = self._get_args()
        opt.update(OPT)
        manager = MessengerManager(opt)
        agent = MessengerAgent(opt, manager, AGENT_ID, TASK_ID, PAGE_ID)

        self.assertEqual(agent.data, {'allow_images': False})
        agent.data = {'second_arg': False}
        self.assertEqual(agent.data, {'allow_images': False, 'second_arg': False})
        agent.data['allow_images'] = True
        self.assertEqual(agent.data, {'allow_images': True, 'second_arg': False}),
        agent.data = {'third_arg': 1, 'second_arg': True}
        self.assertEqual(
            agent.data, {'allow_images': True, 'second_arg': True, 'third_arg': 1}
        )


if __name__ == "__main__":
    unittest.main()

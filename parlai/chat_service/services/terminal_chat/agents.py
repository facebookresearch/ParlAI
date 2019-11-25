#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.chat_service.core.agents import ChatServiceAgent
import logging

class TerminalAgents(ChatServiceAgent):
    def __init__(self, opt, manager, receiver_id, task_id):
        super().__init__(opt, manager, receiver_id, task_id)
        self.action_id = 1
        self.message_partners = []

    def observe(self, act):
        logging.info(f'Sending new message: {act}')
        quick_replies = act.get('quick_replies', None)
        self.manager.observe_message(self.id, act['text'], quick_replies)

    def put_data(self, message):
        logging.info(f"Received new message: {message}")
        action = {
            'episode_done': False,
            'text': message.get('text', ''),
        }

        self._queue_action(action, self.action_id)
        self.action_id += 1

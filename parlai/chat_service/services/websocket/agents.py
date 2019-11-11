#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from parlai.chat_service.core.agents import ChatServiceAgent


class WebsocketAgent(ChatServiceAgent):
    """Class for a person that can act in a ParlAI world via websockets"""

    def __init__(self, *args):
        super().__init__(*args)
        self.message_partners = []
        self.action_id = 1

    def observe(self, act):
        """Send an agent a message through the websocket manager"""
        logging.info(f"Sending new message: {act}")
        if 'payload' in act:
            raise ValueError("Payload not supported yet by websockets")
        else:
            msg = act['text']
            self.manager.observe_message(self.id, msg)

    def put_data(self, message):
        """Put data into the message queue"""
        logging.info(f"Received new message: {message}")
        action = {'episode_done': False, 'text': message['text']}
        self._queue_action(action, self.action_id)
        self.action_id += 1

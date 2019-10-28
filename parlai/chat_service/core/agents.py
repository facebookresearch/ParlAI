#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from queue import Queue
from parlai.core.agents import Agent


class ChatServiceAgent(Agent):
    """Base class for a person on a chat serivce that can act in a ParlAI world"""

    def __init__(self, opt, manager, id):
        super().__init__(opt)
        self.manager = manager
        self.id = id

        self.acted_packets = {}
        self.data = {}
        self.disp_id = 'NewUser'
        self.msg_queue = Queue()
        self.observed_packets = {}

        self.stored_data = {}
        # initialize stored data
        self.set_stored_data()

    def observe(self, act):
        """Send an agent a message through the manager"""
        pass

    def put_data(self, message):
        """Put data into the message queue if it hasn't already been seen"""
        pass

    def set_stored_data(self):
        """Gets agent state data from manager"""
        agent_state = self.manager.get_agent_state(self.id)
        if agent_state is not None and hasattr(agent_state, 'stored_data'):
            self.stored_data = agent_state.stored_data

    def get_new_act_message(self):
        """Get a new act message if one exists, return None otherwise"""
        if not self.msg_queue.empty():
            return self.msg_queue.get()
        return None

    def act(self):
        """Pulls a message from the message queue. If none exist returns None."""
        msg = self.get_new_act_message()
        return msg

    def act_blocking(self):
        """Repeatedly loop until we retrieve a message from the queue."""
        while True:
            msg = self.act()
            if msg is not None:
                return msg
            time.sleep(0.2)

    def episode_done(self):
        """Return whether or not this agent believes the conversation to
        be done"""
        pass

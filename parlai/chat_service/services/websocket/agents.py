#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from queue import Queue

from parlai.core.agents import Agent


class WebsocketAgent(Agent):
    """Class for a person that can act in a ParlAI world via websockets"""

    def __init__(self, opt, manager, task_id, socketID):
        super().__init__(opt)
        self.manager = manager
        self.id = socketID

        self.active = True
        self.msg_queue = Queue()
        self.stored_data = {}
        self.set_stored_data()

    def set_stored_data(self):
        """Gets agent state data from manager"""
        agent_state = self.manager.get_agent_state(self.id)
        if agent_state is not None and hasattr(agent_state, 'stored_data'):
            self.stored_data = agent_state.stored_data

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import logging
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
        self.task_id = task_id
        self.message_request_time = None
        self.set_stored_data()

    def set_stored_data(self):
        """Gets agent state data from manager"""
        agent_state = self.manager.get_agent_state(self.id)
        if agent_state is not None and hasattr(agent_state, 'stored_data'):
            self.stored_data = agent_state.stored_data

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
        action = {'episode_done': False, 'text': message}
        self.msg_queue.put(action)

    def get_new_act_message(self):
        """Get a new act message if one exists, return None otherwise"""
        # Check if person has sent a message
        if not self.msg_queue.empty():
            return self.msg_queue.get()

        return None

    def mark_inactive(self):
        # some kind of behavior to send a message when a user is marked as
        # being inactive. Could be useful. Should return a message to be sent
        pass

    def act(self, timeout=None):
        """Pulls a message from the message queue. If none exist returns None
        unless the timeout has expired.
        """
        # if this is the first act since last sent message start timing
        if self.message_request_time is None:
            self.message_request_time = time.time()

        # If checking timeouts
        if timeout:
            # If time is exceeded, timeout
            if time.time() - self.message_request_time > timeout:
                return self.mark_inactive()

        # Get a new message, if it's not None reset the timeout
        msg = self.get_new_act_message()
        if msg is not None:
            # Do not allow agent to send empty strings
            if msg == "":
                msg = None

            if msg is not None and self.message_request_time is not None:
                self.message_request_time = None
        return msg

    def act_blocking(self, timeout=None):
        """Repeatedly loop until we retrieve a message from the queue"""
        while True:
            msg = self.act(timeout=timeout)
            if msg is not None:
                return msg
            time.sleep(0.2)

    def episode_done(self):
        """Return whether or not this agent believes the conversation to
        be done"""
        return self.manager.shutting_down

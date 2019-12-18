#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from abc import ABC, abstractmethod
from queue import Queue
from parlai.core.agents import Agent


class ChatServiceAgent(Agent, ABC):
    """
    Base class for a person on a chat serivce that can act in a ParlAI world.
    """

    def __init__(self, opt, manager, receiver_id, task_id):
        super().__init__(opt)
        self.manager = manager
        self.id = receiver_id
        self.task_id = task_id
        self.acted_packets = {}
        self.data = {}
        self.msg_queue = Queue()
        self.observed_packets = {}
        self.message_request_time = None
        self.stored_data = {}
        self.message_partners = []
        # initialize stored data
        self.set_stored_data()

    @abstractmethod
    def observe(self, act):
        """
        Send an agent a message through the manager.
        """
        pass

    def _send_payload(self, receiver_id, data, quick_replies=None, persona_id=None):
        """
        Send a payload through the message manager.

        :param receiver_id:
            int identifier for agent to send message to
        :param data:
            object data to send
        :param quick_replies:
            list of quick replies
        :param persona_id:
            identifier of persona
        :return:
            a dictionary of a json response from the manager observing a payload
        """
        return self.manager.observe_payload(
            receiver_id, data, quick_replies, persona_id
        )

    @abstractmethod
    def put_data(self, message):
        """
        Put data into the message queue if it hasn't already been seen.
        """
        pass

    def _queue_action(self, action, act_id, act_data=None):
        """
        Add an action to the queue with given id and info if it hasn't already been
        seen.

        :param action:
            action to be added to message queue
        :param act_id:
            an identifier to check if the action has been seen or to
            mark the action as seen
        :param act_data:
            any data about the given action you may want to record when
            marking it as seen
        """
        if act_id not in self.acted_packets:
            self.acted_packets[act_id] = act_data
            self.msg_queue.put(action)

    def set_stored_data(self):
        """
        Gets agent state data from manager.
        """
        agent_state = self.manager.get_agent_state(self.id)
        if agent_state is not None and hasattr(agent_state, 'stored_data'):
            self.stored_data = agent_state.stored_data

    def get_new_act_message(self):
        """
        Get a new act message if one exists, return None otherwise.
        """
        if not self.msg_queue.empty():
            return self.msg_queue.get()
        return None

    def act(self):
        """
        Pulls a message from the message queue.

        If none exist returns None.
        """
        msg = self.get_new_act_message()
        return msg

    def _check_timeout(self, timeout=None):
        """
        Return whether enough time has passed than the timeout amount.
        """
        if timeout:
            return time.time() - self.message_request_time > timeout
        return False

    def act_blocking(self, timeout=None):
        """
        Repeatedly loop until we retrieve a message from the queue.
        """
        while True:
            if self.message_request_time is None:
                self.message_request_time = time.time()
            msg = self.act()
            if msg is not None:
                self.message_request_time = None
                return msg
            if self._check_timeout(timeout):
                return None
            time.sleep(0.2)

    def episode_done(self):
        """
        Return whether or not this agent believes the conversation to be done.
        """
        return self.manager.shutting_down

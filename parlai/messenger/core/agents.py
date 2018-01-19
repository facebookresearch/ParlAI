# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import time
from queue import Queue

from parlai.core.agents import Agent
import parlai.messenger.core.data_model as data_model


class MessengerAgent(Agent):
    """Base class for a person on messenger that can act in a ParlAI world"""

    def __init__(self, opt, manager, assignment_id, messenger_psid, page_id):
        super().__init__(opt)
        self.manager = manager
        self.id = messenger_psid
        self.assignment_id = assignment_id
        self.active = True
        self.message_request_time = None
        self.recieved_packets = {}
        self.msg_queue = Queue()

    def observe(self, msg):
        """Send an agent a message through the mturk manager"""
        self.manager.send_message(self.id, self.assignment_id, msg)

    def put_data(self, id, data):
        """Put data into the message queue if it hasn't already been seen"""
        if id not in self.recieved_packets:
            self.recieved_packets[id] = True
            self.msg_queue.put(data)

    def get_new_act_message(self):
        """Get a new act message if one exists, return None otherwise"""
        # Check if person has sent a message
        if not self.msg_queue.empty():
            msg = self.msg_queue.get()
            if msg['id'] == self.id:
                return msg

        # There are no messages to be sent
        if not self.active:
            # TODO return a status that notes the user is
            # not active right now
            pass

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
        if msg is not None and self.message_request_time is not None:
            self.message_request_time = None
        return msg

    def episode_done(self):
        """Return whether or not this agent believes the conversation to
        be done"""
        return self.manager.assignment_done(self.assignment_id)

    def shutdown(self, timeout=None, direct_submit=False):
        """Shuts down a hit when it is completed"""
        # Timeout in seconds, after which the HIT will be expired automatically
        command_to_send = data_model.COMMAND_SHOW_DONE_BUTTON
        if direct_submit:
            command_to_send = data_model.COMMAND_SUBMIT_HIT
        if not (self.hit_is_abandoned or self.hit_is_returned or
                self.disconnected or self.hit_is_expired):
            self.manager.mark_workers_done([self])
            self.manager.send_command(
                self.worker_id,
                self.assignment_id,
                {'text': command_to_send},
            )
            return self.wait_for_hit_completion(timeout=timeout)

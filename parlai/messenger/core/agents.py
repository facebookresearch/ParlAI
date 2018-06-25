# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import time
from queue import Queue

from parlai.core.agents import Agent


class MessengerAgent(Agent):
    """Base class for a person on messenger that can act in a ParlAI world"""

    def __init__(self, opt, manager, task_id, messenger_psid, page_id):
        super().__init__(opt)
        self.manager = manager
        self.id = messenger_psid
        self.disp_id = 'NewUser'
        self.task_id = task_id
        self.active = True
        self.message_request_time = None
        self.acted_packets = {}
        self.observed_packets = {}
        self.msg_queue = Queue()
        self.stored_data = {}
        self.message_partners = []
        # initialize stored data
        self.set_stored_data()

    def observe(self, act):
        """Send an agent a message through the mturk manager"""
        if 'payload' in act:
            resp = self.manager.observe_payload(
                self.id,
                act['payload'],
                act.get('quick_replies', None),
            )
        else:
            if act['id'] != '':
                msg = '{}: {}'.format(act['id'], act['text'])
            else:
                msg = act['text']
            resp = self.manager.observe_message(
                self.id, msg,
                act.get('quick_replies', None),
            )
        try:
            mid = resp[0]['message_id']
            if mid not in self.observed_packets:
                self.observed_packets[mid] = act
        except Exception:
            print(
                '{} could not be extracted to an observed message'.format(resp)
            )

    def observe_typing_on(self):
        """Allow agent to observe typing indicator"""
        self.manager.message_sender.typing_on(self.id)

    def put_data(self, message):
        """Put data into the message queue if it hasn't already been seen"""
        mid = message['message']['mid']
        seq = message['message'].get('seq', None)
        if 'text' not in message['message']:
            print('Msg: {} could not be extracted to text format'.format(
                message['message']))
            return
        text = message['message']['text']
        if text is None:
            text = message['message']['payload']
        if mid not in self.acted_packets:
            self.acted_packets[mid] = {
                'mid': mid,
                'seq': seq,
                'text': text,
            }
            action = {
                'episode_done': False,
                'text': text,
                'id': self.disp_id,
            }
            self.msg_queue.put(action)

    def set_stored_data(self):
        """Gets agent state data from manager"""
        agent_state = self.manager.get_agent_state(self.id)
        if agent_state is not None:
            self.stored_data = agent_state.stored_data

    def get_new_act_message(self):
        """Get a new act message if one exists, return None otherwise"""
        # Check if person has sent a message
        if not self.msg_queue.empty():
            return self.msg_queue.get()

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
        return self.manager.shutting_down

    def shutdown(self):
        pass

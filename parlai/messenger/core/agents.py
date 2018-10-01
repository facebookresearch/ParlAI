#!/usr/bin/env python3

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

        self.acted_packets = {}
        self.active = True
        self.data = {}
        self.disp_id = 'NewUser'
        self.message_partners = []
        self.message_request_time = None
        self.msg_queue = Queue()
        self.observed_packets = {}
        self.page_id = page_id
        self.task_id = task_id

        self.stored_data = {}
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
        text = message['message'].get('text')
        if text is None:
            text = message['message']['payload']
        img_attempt = True if 'image' in message['message'] else False
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
                'sticker_sender': message.get('sticker_sender', None),
                'img_attempt': img_attempt,
            }
            self.msg_queue.put(action)

    def set_stored_data(self):
        """Gets agent state data from manager"""
        agent_state = self.manager.get_agent_state(self.id)
        if agent_state is not None and hasattr(agent_state, 'stored_data'):
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
        if msg is not None:
            if msg.get('img_attempt'):
                # Let agent know that they cannot send images if they
                # attempted to send one
                msg = None
                act = {'id': 'SYSTEM',
                       'text': 'Only text messages are supported at this time. '
                               'Please try with a text-only message.',
                       'episode_done': True}
                self.observe(act)
            elif not msg.get('text'):
                # Do not allow agent to send empty strings
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

    def shutdown(self):
        pass

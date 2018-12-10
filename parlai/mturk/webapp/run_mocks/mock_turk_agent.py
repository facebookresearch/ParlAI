#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import time
from queue import Queue

from parlai.core.agents import Agent
from parlai.mturk.core.agents import AssignState
import parlai.mturk.core.shared_utils as shared_utils

# Special act messages for failure states
MTURK_DISCONNECT_MESSAGE = '[DISCONNECT]'  # Turker disconnected from conv
TIMEOUT_MESSAGE = '[TIMEOUT]'  # the Turker did not respond but didn't return
RETURN_MESSAGE = '[RETURNED]'  # the Turker returned the HIT


class MockTurkAgent(Agent):
    """Mock turk agent that can act in a parlai mturk world"""

    # MTurkAgent Possible Statuses
    ASSIGNMENT_NOT_DONE = 'NotDone'
    ASSIGNMENT_DONE = 'Submitted'
    ASSIGNMENT_APPROVED = 'Approved'
    ASSIGNMENT_REJECTED = 'Rejected'

    MTURK_DISCONNECT_MESSAGE = MTURK_DISCONNECT_MESSAGE
    TIMEOUT_MESSAGE = TIMEOUT_MESSAGE
    RETURN_MESSAGE = RETURN_MESSAGE

    def __init__(self, opt, mturk_manager, hit_id, assignment_id, worker_id):
        super().__init__(opt)
        self.conversation_id = None
        self.mturk_manager = mturk_manager
        self.id = None
        self.mock_status = AssignState.STATUS_NONE
        self.state = AssignState()
        self.assignment_id = assignment_id
        self.hit_id = hit_id
        self.worker_id = worker_id
        self.some_agent_disconnected = False
        self.hit_is_expired = False
        self.hit_is_abandoned = False  # state from Amazon MTurk system
        self.hit_is_returned = False  # state from Amazon MTurk system
        self.hit_is_complete = False  # state from Amazon MTurk system
        self.disconnected = False
        self.wants_message = False
        self.task_group_id = mturk_manager.task_group_id
        self.message_request_time = None
        self.creation_time = time.time()
        self.msg_queue = Queue()
        self.unread_messages = []

    def get_update_packet(self):
        """Produce an update packet that represents the state change
        of this agent"""
        send_messages = []
        while len(self.unread_messages) > 0:
            pkt = self.unread_messages.pop(0)
            send_messages.append(pkt.data)
        done_text = None
        if self.state.is_final() and \
                self.get_status() != AssignState.STATUS_DONE:
            done_text = self.state.get_inactive_command_text()[0]
        return {
            'new_messages': send_messages,
            'all_messages': self.state.get_messages(),
            'wants_message': self.wants_message,
            'disconnected': self.disconnected,
            'agent_id': self.id,
            'worker_id': self.worker_id,
            'conversation_id': self.conversation_id,
            'task_done': self.state.is_final(),
            'done_text': done_text,
            'status': self.state.get_status(),
        }

    def set_status(self, status):
        """Set the status of this agent on the task"""
        self.state.set_status(status)

    def get_status(self):
        """Get the status of this agent on its task"""
        return self.state.get_status()

    def submitted_hit(self):
        return self.get_status() in [
            AssignState.STATUS_DONE,
            AssignState.STATUS_PARTNER_DISCONNECT
        ]

    def is_final(self):
        """Determine if this agent is in a final state"""
        return self.state.is_final()

    def append_message(self, message):
        """Add a received message to the state"""
        self.state.append_message(message)

    def set_last_command(self, command):
        """Changes the last command recorded as sent to the agent"""
        self.state.set_last_command(command)

    def get_last_command(self):
        """Returns the last command to be sent to this agent"""
        return self.state.get_last_command()

    def clear_messages(self):
        """Clears the message history for this agent"""
        self.state.clear_messages()

    def get_messages(self):
        """Returns all the messages stored in the state"""
        return self.state.get_messages()

    def get_connection_id(self):
        """Returns an appropriate connection_id for this agent"""
        return "{}_{}".format(self.worker_id, self.assignment_id)

    def log_reconnect(self):
        pass

    def get_inactive_command_data(self):
        """Get appropriate inactive command data to respond to a reconnect"""
        text, command = self.state.get_inactive_command_text()
        return {
            'text': command,
            'inactive_text': text,
            'conversation_id': self.conversation_id,
            'agent_id': self.worker_id,
        }

    def wait_for_status(self, desired_status):
        """Suspend a thread until a particular assignment state changes
        to the desired state
        """
        while True:
            if self.get_status() == desired_status:
                return True
            if self.is_final():
                return False
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)

    def is_in_task(self):
        return self.status == AssignState.STATUS_IN_TASK

    def observe(self, msg):
        """Send an agent a message through the mturk_manager"""
        self.mturk_manager.send_message(
            self.worker_id, self.assignment_id, msg)

    def put_data(self, id, data):
        """Put data into the message queue if it hasn't already been seen"""
        self.msg_queue.put(data)

    def flush_msg_queue(self):
        """Clear all messages in the message queue"""
        if self.msg_queue is None:
            return
        while not self.msg_queue.empty():
            self.msg_queue.get()

    def reduce_state(self):
        """Cleans up resources related to maintaining complete state"""
        self.flush_msg_queue()
        self.msg_queue = None

    def get_new_act_message(self):
        """Get a new act message if one exists, return None otherwise"""
        # See if any agent has disconnected
        if self.disconnected or self.some_agent_disconnected:
            msg = {
                'id': self.id,
                'text': MTURK_DISCONNECT_MESSAGE,
                'episode_done': True
            }
            return msg

        # Check if the current turker already returned the HIT
        if self.hit_is_returned:
            msg = {
                'id': self.id,
                'text': RETURN_MESSAGE,
                'episode_done': True
            }
            return msg

        if self.msg_queue is not None:
            # Check if Turker sends a message
            while not self.msg_queue.empty():
                msg = self.msg_queue.get()
                if msg['id'] == self.id:
                    return msg

        # There are no messages to be sent
        return None

    def prepare_timeout(self):
        """Log a timeout event, tell mturk manager it occurred, return message
        to return for the act call
        """
        shared_utils.print_and_log(
            logging.INFO,
            '{} timed out before sending.'.format(self.id)
        )
        self.timed_out = True
        msg = {
            'id': self.id,
            'text': TIMEOUT_MESSAGE,
            'episode_done': True
        }
        return msg

    def request_message(self):
        if not (self.disconnected or self.some_agent_disconnected or
                self.hit_is_expired):
            self.wants_message = True

    def act(self, timeout=None, blocking=True):
        """Return a message sent by this agent. If blocking, wait for that
        message to come in. If not, return None if no messages are ready
        at the current moment"""
        if not blocking:
            # if this is the first act since last sent message start timing
            if self.message_request_time is None:
                self.request_message()
                self.message_request_time = time.time()

            # If checking timeouts
            if timeout:
                # If time is exceeded, timeout
                if time.time() - self.message_request_time > timeout:
                    return self.prepare_timeout()

            # Get a new message, if it's not None reset the timeout
            msg = self.get_new_act_message()
            if msg is not None and self.message_request_time is not None:
                self.message_request_time = None
                self.wants_message = False
            return msg
        else:
            self.request_message()
            self.message_request_time = time.time()

            # Timeout in seconds, after which the HIT is expired automatically
            if timeout:
                start_time = time.time()

            # Wait for agent's new message
            while True:
                msg = self.get_new_act_message()
                self.message_request_time = None
                if msg is not None:
                    self.wants_message = False
                    return msg

                # Check if the Turker waited too long to respond
                if timeout:
                    current_time = time.time()
                    if (current_time - start_time) > timeout:
                        self.message_request_time = None
                        return self.prepare_timeout()
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)

    def episode_done(self):
        """Return whether or not this agent believes the conversation to
        be done"""
        if self.status == AssignState.STATUS_DONE:
            return False
        else:
            return True

    def approve_work(self):
        print('[mock] Worker {} approved'.format(self.worker_id))

    def reject_work(self, reason='unspecified'):
        print('[mock] Worker {} rejected for reason {}'.format(
            self.worker_id, reason))

    def block_worker(self, reason='unspecified'):
        print('[mock] Worker {} blocked for reason {}'.format(
            self.worker_id, reason))

    def pay_bonus(self, bonus_amount, reason='unspecified'):
        print('[mock] Worker {} bonused {} for reason {}'.format(
            self.worker_id, bonus_amount, reason))

    def email_worker(self, subject, message_text):
        return True

    def set_hit_is_abandoned(self):
        self.hit_is_abandoned = True

    def wait_for_hit_completion(self, timeout=None):
        pass

    def shutdown(self, timeout=None, direct_submit=False):
        pass

    def update_agent_id(self, agent_id):
        """Workaround used to force an update to an agent_id on the front-end
        to render the correct react components for onboarding and waiting
        worlds. Only really used in special circumstances where different
        agents need different onboarding worlds.
        """
        self.id = agent_id

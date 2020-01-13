#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from parlai.mturk.core.agents import MTurkAgent, TIMEOUT_MESSAGE
from parlai.mturk.core.shared_utils import AssignState
import parlai.mturk.core.shared_utils as shared_utils


class MockTurkAgent(MTurkAgent):
    """
    Mock turk agent that can act in a parlai mturk world.
    """

    def __init__(self, opt, mturk_manager, hit_id, assignment_id, worker_id):
        super().__init__(opt, mturk_manager, hit_id, assignment_id, worker_id)
        self.db_logger = None
        self.mock_status = AssignState.STATUS_NONE
        self.wants_message = False
        self.unread_messages = []
        self.timed_out = False

    def get_update_packet(self):
        """
        Produce an update packet that represents the state change of this agent.
        """
        send_messages = []
        while len(self.unread_messages) > 0:
            pkt = self.unread_messages.pop(0)
            send_messages.append(pkt.data)
        done_text = None
        if self.state.is_final() and self.get_status() != AssignState.STATUS_DONE:
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

    def log_reconnect(self):
        """
        We aren't logging behavior in the mock.
        """
        pass

    def is_in_task(self):
        return self.status == AssignState.STATUS_IN_TASK

    def put_data(self, id, data):
        """
        Put data into the message queue.
        """
        self.msg_queue.put(data)

    def flush_msg_queue(self):
        """
        Clear all messages in the message queue.
        """
        if self.msg_queue is None:
            return
        while not self.msg_queue.empty():
            self.msg_queue.get()

    def prepare_timeout(self):
        """
        Log a timeout event, tell mturk manager it occurred, return message to return
        for the act call.
        """
        shared_utils.print_and_log(
            logging.INFO, '{} timed out before sending.'.format(self.id)
        )
        self.timed_out = True
        return self._get_episode_done_msg(TIMEOUT_MESSAGE)

    def request_message(self):
        if not (
            self.disconnected or self.some_agent_disconnected or self.hit_is_expired
        ):
            self.wants_message = True

    def act(self, timeout=None, blocking=True):
        """
        Retrieve an act in the normal expected way (out of the queue), but clear the act
        request if we do end up getting an act.
        """
        gotten_act = super().act(timeout, blocking)
        if gotten_act is not None:
            self.wants_message = False
            gotten_act['episode_done'] = gotten_act.get('episode_done', False)
        return gotten_act

    def episode_done(self):
        """
        Return whether or not this agent believes the conversation to be done.
        """
        if self.get_status() == AssignState.STATUS_DONE:
            return False
        else:
            return True

    def approve_work(self):
        print('[mock] Worker {} approved'.format(self.worker_id))

    def reject_work(self, reason='unspecified'):
        print('[mock] Worker {} rejected for reason {}'.format(self.worker_id, reason))

    def block_worker(self, reason='unspecified'):
        print('[mock] Worker {} blocked for reason {}'.format(self.worker_id, reason))

    def pay_bonus(self, bonus_amount, reason='unspecified'):
        print(
            '[mock] Worker {} bonused {} for reason {}'.format(
                self.worker_id, bonus_amount, reason
            )
        )

    def email_worker(self, subject, message_text):
        return True

    def set_hit_is_abandoned(self):
        self.hit_is_abandoned = True

    def wait_for_hit_completion(self, timeout=None):
        pass

    def shutdown(self, timeout=None, direct_submit=False):
        pass

    def update_agent_id(self, agent_id):
        """
        State is sent directly from the agent, so no need to send like MTurkAgent does
        in the full version.
        """
        self.id = agent_id

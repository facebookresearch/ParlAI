# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


import threading
import time
from queue import Queue
import uuid

from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
from parlai.mturk.core.shared_utils import print_and_log, THREAD_SHORT_SLEEP, \
                                           THREAD_MTURK_POLLING_SLEEP

# TODO-1 move these somewhere that makes more sense
ASSIGNMENT_NOT_DONE = 'NotDone'
ASSIGNMENT_DONE = 'Submitted'
ASSIGNMENT_APPROVED = 'Approved'
ASSIGNMENT_REJECTED = 'Rejected'

# Special act messages for failure states
MTURK_DISCONNECT_MESSAGE = '[DISCONNECT]' # Turker disconnected from conv
TIMEOUT_MESSAGE = '[TIMEOUT]' # the Turker did not respond but didn't return
RETURN_MESSAGE = '[RETURNED]' # the Turker returned the HIT


class MTurkAgent(Agent):
    """Base class for an MTurkAgent that can act in a ParlAI world"""
    def __init__(self, opt, manager, hit_id, assignment_id, worker_id):
        super().__init__(opt)

        self.conversation_id = None
        self.manager = manager
        self.id = None
        self.assignment_id = assignment_id
        self.hit_id = hit_id
        self.worker_id = worker_id
        self.some_agent_disconnected = False
        self.hit_is_abandoned = False
        self.hit_is_expired = False
        self.hit_is_accepted = False # state from Amazon MTurk system
        self.hit_is_returned = False # state from Amazon MTurk system
        self.disconnected = False
        self.task_group_id = manager.task_group_id

        self.msg_queue = Queue()

        # self.check_hit_status_thread = threading.Thread(
        #    target=self._check_hit_status)
        # self.check_hit_status_thread.daemon = True
        # self.check_hit_status_thread.start()

    def _check_hit_status(self):
        """Monitor and update the HIT status by polling"""
        # TODO-1 replace with code that subscribes to notifs to update status
        # Check if HIT is accepted
        while True:
            if self.hit_id:
                response = self.manager.get_hit(hit_id=self.hit_id)
                # Amazon MTurk system acknowledges that the HIT is accepted
                if response['HIT']['NumberOfAssignmentsPending'] == 1:
                    print_and_log(('Worker has accepted the HIT '
                                   '(acknowledged by MTurk API).'), False)
                    self.hit_is_accepted = True
                    break
            time.sleep(THREAD_MTURK_POLLING_SLEEP)
        while True:
            if self.hit_id:
                response = self.manager.get_hit(hit_id=self.hit_id)
                # HIT is returned
                if response['HIT']['NumberOfAssignmentsAvailable'] == 1:
                    self.hit_is_returned = True
                    # If the worker is still in onboarding, then we don't need
                    # to expire the HIT.
                    # If the worker is already in a conversation, then we
                    # should expire the HIT to keep the total number of
                    # available HITs consistent with the number of
                    # conversations left.
                    if self.is_in_task():
                        print_and_log(('Worker has returned the HIT. Since '
                            'the worker is already in a task conversation, '
                            'we are expiring the HIT.'), False)
                        self.manager.expire_hit(hit_id=self.hit_id)
                    else:
                        print_and_log(('Worker has returned the HIT. Since '
                            'the worker is still in onboarding, we will not '
                            'expire the HIT.'), False)
                    # we will not be using this MTurkAgent object for another
                    # worker, so no need to check its status anymore
                    return
            time.sleep(THREAD_MTURK_POLLING_SLEEP)

    def is_in_task(self):
        """Use conversation_id to determine if an agent is in a task"""
        if self.conversation_id:
            return 't_' in self.conversation_id
        return False

    def observe(self, msg):
        """Send an agent a message through the mturk manager"""
        self.manager.send_message(self.worker_id, self.assignment_id, msg)


    def act(self, timeout=None, blocking=True):
        if blocking:
            """Waits for a message to send to other agents in the world"""
            if not (self.disconnected or self.some_agent_disconnected or
                    self.hit_is_expired):
                self.manager.send_command(
                    self.worker_id,
                    self.assignment_id,
                    {'text': data_model.COMMAND_SEND_MESSAGE}
                )

            # Timeout in seconds, after which the HIT will be expired automatically
            if timeout:
                start_time = time.time()

            # Wait for agent's new message
            while True:
                # Check if Turker sends a message
                if not self.msg_queue.empty():
                    msg = self.msg_queue.get()
                    if msg['id'] == self.id:
                        return msg

                if self.disconnected:
                    print("THIS AGENT DISCONNECTED")
                    msg = {
                        'id': self.id,
                        'text': MTURK_DISCONNECT_MESSAGE,
                        'episode_done': True
                    }
                    return msg

                # See if another agent has disconnected
                if self.some_agent_disconnected:
                    print("SOME AGENT DISCONNECTED")
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

                # Check if the Turker waited too long to respond
                if timeout:
                    current_time = time.time()
                    if (current_time - start_time) > timeout:
                        print_and_log('{} is timeout.'.format(self.id), False)
                        self.set_hit_is_abandoned()
                        msg = {
                            'id': self.id,
                            'text': TIMEOUT_MESSAGE,
                            'episode_done': True
                        }
                        return msg
                time.sleep(THREAD_SHORT_SLEEP)
        else:
            if self.some_agent_disconnected:
                return {'id': self.id,
                        'text': MTURK_DISCONNECT_MESSAGE,
                        'episode_done': True}
            elif self.msg_queue.empty():
                return None
            else:
                return self.msg_queue.get()

    def change_conversation(self, conversation_id, agent_id, change_callback):
        """Handle changing a conversation for an agent, takes a callback for
        when the command is acknowledged
        """
        self.id = agent_id
        self.conversation_id = conversation_id
        data = {
            'text': data_model.COMMAND_CHANGE_CONVERSATION,
            'conversation_id': conversation_id,
            'agent_id': agent_id
        }
        self.manager.send_command(
            self.worker_id,
            self.assignment_id,
            data,
            ack_func=change_callback
        )

    def episode_done(self):
        # TODO-2 provide documentation for what this is supposed to be used for
        return False

    def _print_not_available_for(self, item):
        print_and_log(
            'Conversation ID: {}, Agent ID: {} - HIT '
            'is abandoned and thus not available for '
            '{}.'.format(self.conversation_id, self.id, item)
        )

    def approve_work(self):
        """Approving work after it has been submitted"""
        if self.hit_is_abandoned:
            self._print_not_available_for('review')
        else:
            if self.manager.get_agent_work_status(self.assignment_id) == \
                    ASSIGNMENT_DONE:
                self.manager.approve_work(assignment_id=self.assignment_id)
                print_and_log(
                    'Conversation ID: {}, Agent ID: {} - HIT is '
                    'approved.'.format(self.conversation_id, self.id)
                )
            else:
                print_and_log('Cannot approve HIT. Reason: Turker hasn\'t '
                              'completed the HIT yet.')

    def reject_work(self, reason='unspecified'):
        """Reject work after it has been submitted"""
        if self.hit_is_abandoned:
            self._print_not_available_for('review')
        else:
            if self.manager.get_agent_work_status(self.assignment_id) == \
                    ASSIGNMENT_DONE:
                self.manager.reject_work(self.assignment_id, reason)
                print_and_log(
                    'Conversation ID: {}, Agent ID: {} - HIT is '
                    'rejected.'.format(self.conversation_id, self.id)
                )
            else:
                print_and_log('Cannot reject HIT. Reason: Turker hasn\'t '
                              'completed the HIT yet.')

    def block_worker(self, reason='unspecified'):
        """Block a worker from our tasks"""
        self.manager.block_worker(worker_id=self.worker_id, reason=reason)
        print_and_log(
            'Blocked worker ID: {}. Reason: {}'.format(self.worker_id, reason)
        )

    def pay_bonus(self, bonus_amount, reason='unspecified'):
        """Pays the given agent the given bonus"""
        if self.hit_is_abandoned:
            self._print_not_available_for('bonus')
        else:
            if self.manager.get_agent_work_status(self.assignment_id) in \
                    (ASSIGNMENT_DONE, ASSIGNMENT_APPROVED):
                unique_request_token = str(uuid.uuid4())
                self.manager.pay_bonus(
                    worker_id=self.worker_id,
                    bonus_amount=bonus_amount,
                    assignment_id=self.assignment_id,
                    reason=reason,
                    unique_request_token=unique_request_token
                )
            else:
                print_and_log('Cannot pay bonus for HIT. Reason: Turker '
                              'hasn\'t completed the HIT yet.')

    def email_worker(self, subject, message_text):
        """Sends an email to a worker, returns true on a successful send"""
        response = self.manager.email_worker(
            worker_id=self.worker_id,
            subject=subject,
            message_text=message_text
        )
        if 'success' in response:
            print_and_log(
                'Email sent to worker ID: {}: Subject: {}: Text: {}'.format(
                    self.worker_id,
                    subject,
                    message_text
                )
            )
            return True
        elif 'failure' in response:
            print_and_log(
                "Unable to send email to worker ID: {}. Error: {}".format(
                    self.worker_id,
                    response['failure']
                )
            )
            return False

    def set_hit_is_abandoned(self):
        """Update local state to abandoned and expire the hit through MTurk"""
        if not self.hit_is_abandoned:
            self.hit_is_abandoned = True
            self.manager.force_expire_hit(self.worker_id, self.assignment_id)

    def wait_for_hit_completion(self, timeout=None):
        """Waits for a hit to be marked as complete"""
        # Timeout in seconds, after which the HIT will be expired automatically
        if timeout:
            start_time = time.time()
        while self.manager.get_agent_work_status(self.assignment_id) != \
                ASSIGNMENT_DONE:
            # Check if the Turker already returned/disconnected
            if self.hit_is_returned or self.disconnected:
                return False
            if timeout:
                current_time = time.time()
                if (current_time - start_time) > timeout:
                    print_and_log(
                        "Timeout waiting for Turker to complete HIT."
                    )
                    self.set_hit_is_abandoned()
                    return False
            print_and_log('Waiting for ({})_({}) to complete {}...'.format(
                self.worker_id, self.assignment_id, self.conversation_id
            ), False)
            time.sleep(THREAD_MTURK_POLLING_SLEEP)
        print_and_log('Conversation ID: {}, Agent ID: {} - HIT is '
                      'done.'.format(self.conversation_id, self.id))
        self.manager.free_workers([self])
        return True

    def shutdown(self, timeout=None, direct_submit=False):
        """Shuts down a hit when it is completed"""
        # Timeout in seconds, after which the HIT will be expired automatically
        command_to_send = data_model.COMMAND_SHOW_DONE_BUTTON
        if direct_submit:
            command_to_send = data_model.COMMAND_SUBMIT_HIT
        if not (self.hit_is_abandoned or self.hit_is_returned or \
                self.disconnected or self.hit_is_expired):
            self.manager.mark_workers_done([self])
            self.manager.send_command(
                self.worker_id,
                self.assignment_id,
                {'text': command_to_send},
            )
            return self.wait_for_hit_completion(timeout=timeout)

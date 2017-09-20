# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import logging
import threading
import time
from queue import Queue
import uuid

from parlai.core.agents import Agent
from parlai.mturk.core.worker_state import WorkerState, AssignState
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils

# Special act messages for failure states
MTURK_DISCONNECT_MESSAGE = '[DISCONNECT]' # Turker disconnected from conv
TIMEOUT_MESSAGE = '[TIMEOUT]' # the Turker did not respond but didn't return
RETURN_MESSAGE = '[RETURNED]' # the Turker returned the HIT


class MTurkAgent(Agent):
    """Base class for an MTurkAgent that can act in a ParlAI world"""

    # MTurkAgent Possible Statuses
    ASSIGNMENT_NOT_DONE = 'NotDone'
    ASSIGNMENT_DONE = 'Submitted'
    ASSIGNMENT_APPROVED = 'Approved'
    ASSIGNMENT_REJECTED = 'Rejected'

    def __init__(self, opt, manager, hit_id, assignment_id, worker_id):
        super().__init__(opt)

        self.conversation_id = None
        self.manager = manager
        self.id = None
        self.state = AssignState()
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
        self.message_request_time = None

        self.msg_queue = Queue()

        # TODO-1 replace with code that subscribes to notifs to update status
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
                    shared_utils.print_and_log(
                        logging.INFO,
                        'Worker has accepted the HIT'
                    )
                    self.hit_is_accepted = True
                    break
            time.sleep(shared_utils.THREAD_MTURK_POLLING_SLEEP)
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
                        shared_utils.print_and_log(
                            logging.INFO,
                            'Worker {}_{} has returned the HIT {}. Since '
                            'the worker is already in a task conversation, '
                            'we are expiring the HIT.'.format(
                                self.worker_id,
                                self.assignment_id,
                                self.hit_id
                            )
                        )
                        self.manager.expire_hit(hit_id=self.hit_id)
                    else:
                        shared_utils.print_and_log(
                            logging.INFO,
                            'Worker {}_{} has returned the HIT {}. Since '
                            'the worker is still in onboarding, we will not '
                            'expire the HIT.'.format(
                                self.worker_id,
                                self.assignment_id,
                                self.hit_id
                            )
                        )
                    # we will not be using this MTurkAgent object for another
                    # worker, so no need to check its status anymore
                    return
            time.sleep(shared_utils.THREAD_MTURK_POLLING_SLEEP)

    def get_connection_id(self):
        """Returns an appropriate connection_id for this agent"""
        return "{}_{}".format(self.worker_id, self.assignment_id)

    def log_reconnect(self):
        """Log a reconnect of this agent """
        shared_utils.print_and_log(
            logging.DEBUG,
            'Agent ({})_({}) reconnected to {} with status {}'.format(
                self.worker_id, self.assignment_id,
                self.conversation_id, self.state.status
            )
        )

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
            if self.state.status == desired_status:
                break
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)

    def is_in_task(self):
        """Use conversation_id to determine if an agent is in a task"""
        if self.conversation_id:
            return 't_' in self.conversation_id
        return False

    def observe(self, msg):
        """Send an agent a message through the mturk manager"""
        self.manager.send_message(self.worker_id, self.assignment_id, msg)

    def get_new_act_message(self):
        """Get a new act message if one exists, return None otherwise"""
        # Check if Turker sends a message
        if not self.msg_queue.empty():
            msg = self.msg_queue.get()
            if msg['id'] == self.id:
                return msg

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
        self.manager.handle_turker_timeout(
            self.worker_id,
            self.assignment_id
        )
        msg = {
            'id': self.id,
            'text': TIMEOUT_MESSAGE,
            'episode_done': True
        }
        return msg

    def act(self, timeout=None, blocking=True):
        """Sends a message to other agents in the world. If blocking, this
        will wait for the message to come in so it can be sent. Otherwise
        it will return None.
        """
        if not blocking:
            # If checking timeouts
            if timeout:
                # if this is the first act since last sent message start timing
                if self.message_request_time is None:
                    self.message_request_time = time.time()
                # If time is exceeded, timeout
                if time.time() - self.message_request_time > timeout:
                    return self.prepare_timeout()

            # Get a new message, if it's not None reset the timeout
            msg = self.get_new_act_message()
            if msg is not None and self.message_request_time is not None:
                self.message_request_time = None
            return msg
        else:
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
                msg = self.get_new_act_message()
                if msg is not None:
                    return msg

                # Check if the Turker waited too long to respond
                if timeout:
                    current_time = time.time()
                    if (current_time - start_time) > timeout:
                        return self.prepare_timeout()
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)

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
        """Return whether or not this agent believes the conversation to
        be done"""
        if self.manager.get_agent_work_status(self.assignment_id) == \
                self.ASSIGNMENT_NOT_DONE:
            return False
        else:
            return True

    def _print_not_available_for(self, item):
        shared_utils.print_and_log(
            logging.WARN,
            'Conversation ID: {}, Agent ID: {} - HIT '
            'is abandoned and thus not available for '
            '{}.'.format(self.conversation_id, self.id, item),
            should_print=True
        )

    def approve_work(self):
        """Approving work after it has been submitted"""
        if self.hit_is_abandoned:
            self._print_not_available_for('review')
        else:
            if self.manager.get_agent_work_status(self.assignment_id) == \
                    self.ASSIGNMENT_DONE:
                self.manager.approve_work(assignment_id=self.assignment_id)
                shared_utils.print_and_log(
                    logging.INFO,
                    'Conversation ID: {}, Agent ID: {} - HIT is '
                    'approved.'.format(self.conversation_id, self.id)
                )
            else:
                shared_utils.print_and_log(
                    logging.WARN,
                    'Cannot approve HIT. Turker hasn\'t completed the HIT yet.'
                )

    def reject_work(self, reason='unspecified'):
        """Reject work after it has been submitted"""
        if self.hit_is_abandoned:
            self._print_not_available_for('review')
        else:
            if self.manager.get_agent_work_status(self.assignment_id) == \
                    self.ASSIGNMENT_DONE:
                self.manager.reject_work(self.assignment_id, reason)
                shared_utils.print_and_log(
                    logging.INFO,
                    'Conversation ID: {}, Agent ID: {} - HIT is '
                    'rejected.'.format(self.conversation_id, self.id)
                )
            else:
                shared_utils.print_and_log(
                    logging.WARN,
                    'Cannot reject HIT. Turker hasn\'t completed the HIT yet.'
                )

    def block_worker(self, reason='unspecified'):
        """Block a worker from our tasks"""
        self.manager.block_worker(worker_id=self.worker_id, reason=reason)
        shared_utils.print_and_log(
            logging.WARN,
            'Blocked worker ID: {}. Reason: {}'.format(self.worker_id, reason),
            should_print=True
        )

    def pay_bonus(self, bonus_amount, reason='unspecified'):
        """Pays the given agent the given bonus"""
        if self.hit_is_abandoned:
            self._print_not_available_for('bonus')
        else:
            if self.manager.get_agent_work_status(self.assignment_id) in \
                    (self.ASSIGNMENT_DONE, self.ASSIGNMENT_APPROVED):
                unique_request_token = str(uuid.uuid4())
                self.manager.pay_bonus(
                    worker_id=self.worker_id,
                    bonus_amount=bonus_amount,
                    assignment_id=self.assignment_id,
                    reason=reason,
                    unique_request_token=unique_request_token
                )
            else:
                shared_utils.print_and_log(
                    logging.WARN,
                    'Cannot pay bonus for HIT. Reason: Turker '
                    'hasn\'t completed the HIT yet.'
                )

    def email_worker(self, subject, message_text):
        """Sends an email to a worker, returns true on a successful send"""
        response = self.manager.email_worker(
            worker_id=self.worker_id,
            subject=subject,
            message_text=message_text
        )
        if 'success' in response:
            shared_utils.print_and_log(
                logging.INFO,
                'Email sent to worker ID: {}: Subject: {}: Text: {}'.format(
                    self.worker_id,
                    subject,
                    message_text
                )
            )
            return True
        elif 'failure' in response:
            shared_utils.print_and_log(
                logging.WARN,
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
            if timeout < 0:
                # Negative timeout is for testing
                self.manager.free_workers([self])
                return True
            start_time = time.time()
        while self.manager.get_agent_work_status(self.assignment_id) != \
                self.ASSIGNMENT_DONE:
            # Check if the Turker already returned/disconnected
            if self.hit_is_returned or self.disconnected:
                return False
            if timeout:
                current_time = time.time()
                if (current_time - start_time) > timeout:
                    shared_utils.print_and_log(
                        logging.INFO,
                        "Timeout waiting for ({})_({}) to complete {}.".format(
                            self.worker_id,
                            self.assignment_id,
                            self.conversation_id
                        )
                    )
                    self.set_hit_is_abandoned()
                    return False
            shared_utils.print_and_log(
                logging.DEBUG,
                'Waiting for ({})_({}) to complete {}...'.format(
                    self.worker_id, self.assignment_id, self.conversation_id
                )
            )
            time.sleep(shared_utils.THREAD_MTURK_POLLING_SLEEP)
        shared_utils.print_and_log(
            logging.INFO,
            'Conversation ID: {}, Agent ID: {} - HIT is done.'.format(
                self.conversation_id, self.id
            )
        )
        self.manager.free_workers([self])
        return True

    def reduce_state(self):
        """Cleans up resources related to maintaining complete state"""
        self.msg_queue = None
        self.state.clear_messages()

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

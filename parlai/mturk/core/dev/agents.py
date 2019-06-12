#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from queue import Queue
import uuid

from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils

# Special act messages for failure states
MTURK_DISCONNECT_MESSAGE = '[DISCONNECT]'  # Turker disconnected from conv
TIMEOUT_MESSAGE = '[TIMEOUT]'  # the Turker did not respond but didn't return
RETURN_MESSAGE = '[RETURNED]'  # the Turker returned the HIT

# TODO move time management into another class, this way we can handle it
# relative to heartbeats. This will come with more thorough testing.


class AssignState():
    """Class for holding state information about an assignment currently
    claimed by an agent
    """

    # Possible Assignment Status Values
    STATUS_NONE = 'none'
    STATUS_ONBOARDING = 'onboarding'
    STATUS_WAITING = 'waiting'
    STATUS_IN_TASK = 'in task'
    STATUS_DONE = 'done'
    STATUS_DISCONNECT = 'disconnect'
    STATUS_PARTNER_DISCONNECT = 'partner disconnect'
    STATUS_PARTNER_DISCONNECT_EARLY = 'partner disconnect early'
    STATUS_EXPIRED = 'expired'
    STATUS_RETURNED = 'returned'
    STATUS_STATIC = 'static'

    def __init__(self, status=None):
        """Create an AssignState to track the state of an agent's assignment"""
        if status is None:
            status = self.STATUS_NONE
        self.status = status
        self.messages = []
        self.last_command = None
        self.message_ids = []

    def clear_messages(self):
        self.messages = []
        self.message_ids = []
        self.last_command = None

    def append_message(self, message):
        """Appends a message to the list of messages, ensures that it is
        not a duplicate message.
        """
        if message['message_id'] in self.message_ids:
            return
        self.message_ids.append(message['message_id'])
        self.messages.append(message)

    def set_last_command(self, command):
        self.last_command = command

    def get_last_command(self):
        return self.last_command

    def get_messages(self):
        return self.messages

    def set_status(self, status):
        """Set the status of this agent on the task"""
        # TODO log to db
        self.status = status

    def get_status(self):
        """Get the status of this agent on its task"""
        # TODO retrieve from db if not set
        return self.status

    def is_final(self):
        """Return True if the assignment is in a final status that
        can no longer be acted on.
        """
        return (self.status == self.STATUS_DISCONNECT or
                self.status == self.STATUS_DONE or
                self.status == self.STATUS_PARTNER_DISCONNECT or
                self.status == self.STATUS_PARTNER_DISCONNECT_EARLY or
                self.status == self.STATUS_RETURNED or
                self.status == self.STATUS_EXPIRED)

    def get_inactive_command_text(self):
        """Get appropriate inactive command and text to respond to a reconnect
        given the current assignment state

        returns text, command
        """
        command = data_model.COMMAND_INACTIVE_HIT
        text = None
        if self.status == self.STATUS_DISCONNECT:
            text = ('You disconnected in the middle of this HIT and were '
                    'marked as inactive. As these HITs often require real-'
                    'time interaction, it is no longer available for '
                    'completion. Please return this HIT and accept a new one '
                    'if you would like to try again.')
        elif self.status == self.STATUS_DONE:
            command = data_model.COMMAND_INACTIVE_DONE
            text = ('You disconnected after completing this HIT without '
                    'marking it as completed. Please press the done button '
                    'below to finish the HIT.')
        elif self.status == self.STATUS_EXPIRED:
            text = ('You disconnected in the middle of this HIT and the '
                    'HIT expired before you reconnected. It is no longer '
                    'available for completion. Please return this HIT and '
                    'accept a new one if you would like to try again.')
        elif self.status == self.STATUS_PARTNER_DISCONNECT:
            command = data_model.COMMAND_INACTIVE_DONE
            text = ('One of your partners disconnected in the middle of the '
                    'HIT. We won\'t penalize you for their disconnect, so '
                    'please use the button below to mark the HIT as complete.')
        elif self.status == self.STATUS_PARTNER_DISCONNECT_EARLY:
            command = data_model.COMMAND_INACTIVE_HIT
            text = ('One of your partners disconnected in the middle of the '
                    'HIT. We won\'t penalize you for their disconnect, but you'
                    ' did not complete enough of the task to submit the HIT. '
                    'Please return this HIT and accept a new one if you would '
                    'like to try again.')
        elif self.status == self.STATUS_RETURNED:
            text = ('You disconnected from this HIT and then returned '
                    'it. As we have marked the HIT as returned, it is no '
                    'longer available for completion. Please accept a new '
                    'HIT if you would like to try again')
        else:
            # We shouldn't be getting an inactive command for the other
            # states so consider this a server error
            text = ('Our server was unable to handle your reconnect properly '
                    'and thus this HIT no longer seems available for '
                    'completion. Please try to connect again or return this '
                    'HIT and accept a new one.')

        return text, command


class MTurkAgent(Agent):
    """Base class for an MTurkAgent that can act in a ParlAI world"""

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
        self.db_logger = mturk_manager.db_logger
        self.id = None
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
        self.task_group_id = mturk_manager.task_group_id
        self.message_request_time = None
        self.recieved_packets = {}
        self.creation_time = time.time()
        self.alived = True  # Used for restoring state after refresh
        self.feedback = None

        self.msg_queue = Queue()

    def _get_episode_done_msg(self, text):
        return {
            'id': self.id,
            'text': text,
            'episode_done': True
        }

    def set_status(self, status):
        """Set the status of this agent on the task, update db"""
        self.state.set_status(status)
        self.mturk_manager.send_state_change(
            self.worker_id,
            self.assignment_id,
            {'agent_status': status},
        )
        if self.db_logger is not None:
            if status == AssignState.STATUS_ONBOARDING:
                self.db_logger.log_start_onboard(
                    self.worker_id, self.assignment_id, self.conversation_id)
            elif status == AssignState.STATUS_WAITING:
                self.db_logger.log_finish_onboard(
                    self.worker_id, self.assignment_id)
            elif status == AssignState.STATUS_IN_TASK:
                self.db_logger.log_start_task(
                    self.worker_id, self.assignment_id, self.conversation_id)
            elif status == AssignState.STATUS_DONE:
                self.db_logger.log_complete_assignment(
                    self.worker_id, self.assignment_id,
                    time.time() + self.mturk_manager.auto_approve_delay,
                    status)
            elif status == AssignState.STATUS_PARTNER_DISCONNECT:
                self.db_logger.log_complete_assignment(
                    self.worker_id, self.assignment_id,
                    time.time() + self.mturk_manager.auto_approve_delay,
                    status)
            elif status == AssignState.STATUS_PARTNER_DISCONNECT_EARLY:
                self.db_logger.log_complete_assignment(
                    self.worker_id, self.assignment_id,
                    time.time() + self.mturk_manager.auto_approve_delay,
                    status)
            elif status == AssignState.STATUS_DISCONNECT:
                self.db_logger.log_disconnect_assignment(
                    self.worker_id, self.assignment_id,
                    time.time() + self.mturk_manager.auto_approve_delay,
                    status)
            elif status == AssignState.STATUS_EXPIRED:
                self.db_logger.log_complete_assignment(
                    self.worker_id, self.assignment_id,
                    time.time() + self.mturk_manager.auto_approve_delay,
                    status)
            elif status == AssignState.STATUS_RETURNED:
                self.db_logger.log_abandon_assignment(
                    self.worker_id, self.assignment_id)

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
        """Log a reconnect of this agent """
        shared_utils.print_and_log(
            logging.DEBUG,
            'Agent ({})_({}) reconnected to {} with status {}'.format(
                self.worker_id, self.assignment_id,
                self.conversation_id, self.get_status()
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
            if self.get_status() == desired_status:
                return True
            if self.is_final():
                return False
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)

    def is_in_task(self):
        """Use conversation_id to determine if an agent is in a task"""
        if self.conversation_id:
            return 't_' in self.conversation_id
        return False

    def observe(self, msg):
        """Send an agent a message through the mturk_manager"""
        self.mturk_manager.send_message(
            self.worker_id, self.assignment_id, msg)

    def put_data(self, id, data):
        """Put data into the message queue if it hasn't already been seen"""
        if id not in self.recieved_packets:
            self.recieved_packets[id] = True
            self.msg_queue.put(data)

    def flush_msg_queue(self):
        """Clear all messages in the message queue. Return flushed messages"""
        messages = []
        if self.msg_queue is None:
            return []
        while not self.msg_queue.empty():
            messages.append(self.msg_queue.get())
        return messages

    def reduce_state(self):
        """Cleans up resources related to maintaining complete state"""
        self.flush_msg_queue()
        self.msg_queue = None
        self.recieved_packets = None

    def get_new_act_message(self):
        """Get a new act message if one exists, return None otherwise"""
        # See if any agent has disconnected
        if self.disconnected or self.some_agent_disconnected:
            return self._get_episode_done_msg(MTURK_DISCONNECT_MESSAGE)

        # Check if the current turker already returned the HIT
        if self.hit_is_returned:
            return self._get_episode_done_msg(RETURN_MESSAGE)

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
        self.mturk_manager.handle_turker_timeout(
            self.worker_id,
            self.assignment_id
        )
        return self._get_episode_done_msg(TIMEOUT_MESSAGE)

    def request_message(self):
        if not (self.disconnected or self.some_agent_disconnected or
                self.hit_is_expired):
            self.mturk_manager.send_command(
                self.worker_id,
                self.assignment_id,
                {'text': data_model.COMMAND_SEND_MESSAGE}
            )

    def act(self, timeout=None, blocking=True):
        """Sends a message to other agents in the world. If blocking, this
        will wait for the message to come in so it can be sent. Otherwise
        it will return None.
        """
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
        if self.mturk_manager.get_agent_work_status(self.assignment_id) == \
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
            if self.mturk_manager.get_agent_work_status(self.assignment_id) \
                    == self.ASSIGNMENT_DONE:
                self.mturk_manager.approve_work(
                    assignment_id=self.assignment_id)
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
            if self.mturk_manager.get_agent_work_status(self.assignment_id) \
                    == self.ASSIGNMENT_DONE:
                self.mturk_manager.reject_work(self.assignment_id, reason)
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
        self.mturk_manager.block_worker(
            worker_id=self.worker_id, reason=reason)
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
            if self.mturk_manager.get_agent_work_status(self.assignment_id) in\
                    (self.ASSIGNMENT_DONE, self.ASSIGNMENT_APPROVED):
                unique_request_token = str(uuid.uuid4())
                self.mturk_manager.pay_bonus(
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
        response = self.mturk_manager.email_worker(
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
        """Update local state to abandoned and mark the HIT as expired"""
        if not self.hit_is_abandoned:
            self.hit_is_abandoned = True
            self.mturk_manager.force_expire_hit(
                self.worker_id, self.assignment_id)

    def wait_completion_timeout(self, iterations):
        """Suspends the thread waiting for hit completion for some number of
        iterations on the THREAD_MTURK_POLLING_SLEEP time"""

        # Determine number of sleep iterations for the amount of time
        # we want to wait before syncing with MTurk. Start with 10 seconds
        # of waiting
        iters = (shared_utils.THREAD_MTURK_POLLING_SLEEP /
                 shared_utils.THREAD_MEDIUM_SLEEP)
        i = 0
        # Wait for the desired number of MTURK_POLLING_SLEEP iterations
        while not self.hit_is_complete and i < iters * iterations:
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)
            i += 1
        return

    def wait_for_hit_completion(self, timeout=None):
        """Waits for a hit to be marked as complete"""
        # Timeout in seconds, after which the HIT will be expired automatically
        if timeout:
            if timeout < 0:
                # Negative timeout is for testing, wait for packet to send
                time.sleep(1)
                self.mturk_manager.free_workers([self])
                return True
            start_time = time.time()
        wait_periods = 1
        self.wait_completion_timeout(wait_periods)
        sync_attempts = 0
        while (
            not self.hit_is_complete and
            self.mturk_manager.get_agent_work_status(self.assignment_id) !=
            self.ASSIGNMENT_DONE
        ):
            if sync_attempts < 8:
                # Scaling on how frequently to poll, doubles time waited on
                # every failure
                wait_periods *= 2
                sync_attempts += 1
            else:
                # Okay we've waited for 45 mins and the HIT still isn't up
                self.disconnected = True
            # Check if the Turker already returned/disconnected
            if self.hit_is_returned or self.disconnected:
                self.mturk_manager.free_workers([self])
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
                    self.mturk_manager.free_workers([self])
                    return False
            shared_utils.print_and_log(
                logging.DEBUG,
                'Waiting for ({})_({}) to complete {}...'.format(
                    self.worker_id, self.assignment_id, self.conversation_id
                )
            )
            self.wait_completion_timeout(wait_periods)

        shared_utils.print_and_log(
            logging.INFO,
            'Conversation ID: {}, Agent ID: {} - HIT is done.'.format(
                self.conversation_id, self.id
            )
        )
        self.mturk_manager.free_workers([self])
        return True

    def shutdown(self, timeout=None, direct_submit=False):
        """Shuts down a hit when it is completed"""
        # Timeout in seconds, after which the HIT will be expired automatically
        if not (self.hit_is_abandoned or self.hit_is_returned or
                self.disconnected or self.hit_is_expired):
            self.mturk_manager.mark_workers_done([self])
            if direct_submit:
                self.mturk_manager.send_command(
                    self.worker_id,
                    self.assignment_id,
                    {'text': data_model.COMMAND_SUBMIT_HIT},
                )
            did_complete = self.wait_for_hit_completion(timeout=timeout)
            if did_complete and self.db_logger is not None:
                self.db_logger.log_submit_assignment(
                    self.worker_id, self.assignment_id)
            # Grab feedback message if it happens to exist
            messages = self.flush_msg_queue()
            for m in messages:
                if m['text'] == '[PEER_REVIEW]':
                    self.feedback = m['task_data']
            return did_complete

    def update_agent_id(self, agent_id):
        """Workaround used to force an update to an agent_id on the front-end
        to render the correct react components for onboarding and waiting
        worlds. Only really used in special circumstances where different
        agents need different onboarding worlds.
        """
        self.mturk_manager.worker_manager.change_agent_conversation(
            agent=self,
            conversation_id=self.conversation_id,
            new_agent_id=agent_id,
        )

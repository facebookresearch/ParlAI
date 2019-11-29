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


# types of exceptions thrown when an agent exits the chat. These are thrown
# on a failed MTurkAgent.act call. If one of these is thrown and not handled,
# the world will die and enter cleanup. If you would like your task to be able
# to continue working when an agent leaves, you'll need to catch these errors.
class AbsentAgentError(Exception):
    """
    Exceptions for when an agent leaves a task.
    """

    def __init__(self, message, worker_id, assignment_id):
        self.message = message
        self.worker_id = worker_id
        self.assignment_id = assignment_id


class AgentDisconnectedError(AbsentAgentError):
    """
    Exception for a real disconnect event (no signal)
    """

    def __init__(self, worker_id, assignment_id):
        super().__init__(f'Agent disconnected', worker_id, assignment_id)


class AgentTimeoutError(AbsentAgentError):
    """
    Exception for when a worker doesn't respond in time.
    """

    def __init__(self, timeout, worker_id, assignment_id):
        super().__init__(f'Agent exceeded {timeout}', worker_id, assignment_id)


class AgentReturnedError(AbsentAgentError):
    """
    Exception for an explicit return event (worker returns task)
    """

    def __init__(self, worker_id, assignment_id):
        super().__init__(f'Agent returned HIT', worker_id, assignment_id)


class AssignState:
    """
    Class for holding state information about an assignment currently claimed by an
    agent.
    """

    # Possible Assignment Status Values
    STATUS_NONE = 'none'
    STATUS_ONBOARDING = 'onboarding'
    STATUS_WAITING = 'waiting'
    STATUS_IN_TASK = 'in task'
    STATUS_DONE = 'done'
    STATUS_DISCONNECT = 'disconnect'
    STATUS_PARTNER_DISCONNECT = 'partner disconnect'
    # TODO remove this
    STATUS_PARTNER_DISCONNECT_EARLY = 'partner disconnect early'
    STATUS_EXPIRED = 'expired'
    STATUS_RETURNED = 'returned'
    STATUS_STATIC = 'static'

    def __init__(self, status=None):
        """
        Create an AssignState to track the state of an agent's assignment.
        """
        if status is None:
            status = self.STATUS_NONE
        self.status = status
        self.messages = []
        self.message_ids = []

    def clear_messages(self):
        self.messages = []
        self.message_ids = []

    def append_message(self, message):
        """
        Appends a message to the list of messages, ensures that it is not a duplicate
        message.
        """
        if message['message_id'] in self.message_ids:
            return
        self.message_ids.append(message['message_id'])
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def set_status(self, status):
        """
        Set the status of this agent on the task.
        """
        # TODO log to db
        self.status = status

    def get_status(self):
        """
        Get the status of this agent on its task.
        """
        # TODO retrieve from db if not set
        return self.status

    def is_final(self):
        """
        Return True if the assignment is in a final status that can no longer be acted
        on.
        """
        return (
            self.status == self.STATUS_DISCONNECT
            or self.status == self.STATUS_DONE
            or self.status == self.STATUS_PARTNER_DISCONNECT
            or
            # TODO REMOVE THIS TYPE
            self.status == self.STATUS_PARTNER_DISCONNECT_EARLY
            or self.status == self.STATUS_RETURNED
            or self.status == self.STATUS_EXPIRED
        )


class MTurkAgent(Agent):
    """
    Base class for an MTurkAgent that can act in a ParlAI world.
    """

    # MTurkAgent Possible Statuses
    ASSIGNMENT_NOT_DONE = 'NotDone'
    ASSIGNMENT_DONE = 'Submitted'
    ASSIGNMENT_APPROVED = 'Approved'
    ASSIGNMENT_REJECTED = 'Rejected'

    def __init__(self, opt, mturk_manager, hit_id, assignment_id, worker_id):
        super().__init__(opt)

        # all MTurkManager functions explicitly used by agents extracted here
        self.m_send_state_change = mturk_manager.send_state_change
        self.m_send_message = mturk_manager.send_message
        self.m_send_command = mturk_manager.send_command
        self.m_approve_work = mturk_manager.approve_work
        self.m_reject_work = mturk_manager.reject_work
        self.m_block_worker = mturk_manager.block_worker
        self.m_soft_block_worker = mturk_manager.soft_block_worker
        self.m_pay_bonus = mturk_manager.pay_bonus
        self.m_email_worker = mturk_manager.email_worker
        self.m_free_workers = mturk_manager.free_workers
        self.m_mark_workers_done = mturk_manager.mark_workers_done
        self.m_force_expire_hit = mturk_manager.force_expire_hit
        self.m_register_worker = mturk_manager.worker_manager.register_to_conv

        # MTurkManager variables used by agents extracted here
        self.auto_approve_delay = mturk_manager.auto_approve_delay

        self.db_logger = mturk_manager.db_logger
        self.task_group_id = mturk_manager.task_group_id

        # Initial state for the agent
        self.conversation_id = None
        self.id = None
        self.state = AssignState()
        self.assignment_id = assignment_id
        self.hit_id = hit_id
        self.worker_id = worker_id
        self.some_agent_disconnected = False
        self.hit_is_expired = False
        self.hit_is_abandoned = False  # state from MTurk SNS system
        self.hit_is_returned = False  # state from MTurk SNS system
        self.hit_is_complete = False  # submission post came through
        self.disconnected = False
        self.message_request_time = None
        self.recieved_packets = {}
        self.creation_time = time.time()
        self.feedback = None
        self.msg_queue = Queue()
        self.completed_message = None

    def set_status(self, status, conversation_id=None, agent_id=None):
        """
        Set the status of this agent on the task, update db, push update to the router.
        """
        self.state.set_status(status)
        update_packet = {'agent_status': status}
        if conversation_id is not None:
            update_packet['conversation_id'] = conversation_id
            self.conversation_id = conversation_id
            self.m_register_worker(self, conversation_id)
        if agent_id is not None:
            self.id = agent_id
            update_packet['agent_id'] = agent_id
        self.m_send_state_change(self.worker_id, self.assignment_id, update_packet)
        if self.db_logger is not None:
            if status == AssignState.STATUS_ONBOARDING:
                self.db_logger.log_start_onboard(
                    self.worker_id, self.assignment_id, self.conversation_id
                )
            elif status == AssignState.STATUS_WAITING:
                self.db_logger.log_finish_onboard(self.worker_id, self.assignment_id)
            elif status == AssignState.STATUS_IN_TASK:
                self.db_logger.log_start_task(
                    self.worker_id, self.assignment_id, self.conversation_id
                )
            elif status == AssignState.STATUS_DONE:
                self.db_logger.log_complete_assignment(
                    self.worker_id,
                    self.assignment_id,
                    time.time() + self.auto_approve_delay,
                    status,
                )
            elif status == AssignState.STATUS_PARTNER_DISCONNECT:
                self.db_logger.log_complete_assignment(
                    self.worker_id,
                    self.assignment_id,
                    time.time() + self.auto_approve_delay,
                    status,
                )
            elif status == AssignState.STATUS_PARTNER_DISCONNECT_EARLY:
                self.db_logger.log_complete_assignment(
                    self.worker_id,
                    self.assignment_id,
                    time.time() + self.auto_approve_delay,
                    status,
                )
            elif status == AssignState.STATUS_DISCONNECT:
                self.db_logger.log_disconnect_assignment(
                    self.worker_id,
                    self.assignment_id,
                    time.time() + self.auto_approve_delay,
                    status,
                )
            elif status == AssignState.STATUS_EXPIRED:
                self.db_logger.log_complete_assignment(
                    self.worker_id,
                    self.assignment_id,
                    time.time() + self.auto_approve_delay,
                    status,
                )
            elif status == AssignState.STATUS_RETURNED:
                self.db_logger.log_abandon_assignment(
                    self.worker_id, self.assignment_id
                )

    def get_status(self):
        """
        Get the status of this agent on its task.
        """
        return self.state.get_status()

    def submitted_hit(self):
        return self.get_status() in [
            AssignState.STATUS_DONE,
            AssignState.STATUS_PARTNER_DISCONNECT,
        ]

    def is_final(self):
        """
        Determine if this agent is in a final state.
        """
        return self.state.is_final()

    # TODO re-examine message clearing
    def clear_messages(self):
        """
        Clears the message history for this agent.
        """
        self.state.clear_messages()

    def get_messages(self):
        """
        Returns all the messages stored in the state.
        """
        return self.state.get_messages()

    def get_connection_id(self):
        """
        Returns an appropriate connection_id for this agent.
        """
        return "{}_{}".format(self.worker_id, self.assignment_id)

    def is_in_task(self):
        """
        Simple check for an agent being in a task.
        """
        return self.get_status() == AssignState.STATUS_IN_TASK

    def observe(self, msg):
        """
        Send an agent a message through the mturk_manager.
        """
        if 'message_id' not in msg:
            msg['message_id'] = str(uuid.uuid4())
        self.state.append_message(msg)
        self.m_send_message(self.worker_id, self.assignment_id, msg)

    def put_data(self, id, data):
        """
        Put data into the message queue if it hasn't already been seen.
        """
        if 'message_id' not in data:
            data['message_id'] = id
        if id not in self.recieved_packets:
            self.state.append_message(data)  # append to message history
            self.recieved_packets[id] = True
            self.msg_queue.put(data)

    def flush_msg_queue(self):
        """
        Clear all messages in the message queue.

        Return flushed messages
        """
        messages = []
        if self.msg_queue is None:
            return []
        while not self.msg_queue.empty():
            messages.append(self.msg_queue.get())
        return messages

    # TODO Only call this from within the agent
    def reduce_state(self):
        """
        Cleans up resources related to maintaining complete state.
        """
        self.flush_msg_queue()
        self.msg_queue = None
        self.recieved_packets = None

    def assert_connected(self):
        """
        Ensures that an agent is still connected.
        """
        # See if any agent has disconnected
        if self.disconnected or self.some_agent_disconnected:
            raise AgentDisconnectedError(self.worker_id, self.assignment_id)

        # Check if the current turker already returned the HIT
        if self.hit_is_returned:
            raise AgentReturnedError(self.worker_id, self.assignment_id)
        return

    def get_new_act_message(self):
        """
        Get a new act message if one exists, return None otherwise.
        """
        self.assert_connected()
        if self.msg_queue is not None:
            # Check if Turker sends a message
            while not self.msg_queue.empty():
                msg = self.msg_queue.get()
                if msg['id'] == self.id:
                    return msg

        # There are no messages to be sent
        return None

    def set_completed_act(self, completed_act):
        """
        Set the completed act for an agent, notes successful submission.
        """
        self.completed_act = completed_act
        self.hit_is_complete = True

    def get_completed_act(self):
        """
        Returns completed act upon arrival, errors on disconnect.
        """
        while self.completed_message is None:
            self.assert_connected()
            time.sleep(shared_utils.THREAD_SHORT_SLEEP)

        return self.completed_message

    def request_message(self):
        if not (
            self.disconnected or self.some_agent_disconnected or self.hit_is_expired
        ):
            self.m_send_command(
                self.worker_id,
                self.assignment_id,
                {'text': data_model.COMMAND_SEND_MESSAGE},
            )

    def act(self, timeout=None, blocking=True):
        """
        Sends a message to other agents in the world.

        If blocking, this will wait for the message to come in so it can be sent.
        Otherwise it will return None.
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
                    raise AgentTimeoutError(timeout, self.worker_id, self.assignment_id)

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
                        raise AgentTimeoutError(
                            timeout, self.worker_id, self.assignment_id
                        )
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)

    def episode_done(self):
        """
        Return whether or not this agent believes the conversation to be done.
        """
        if not self.hit_is_complete:
            return False
        else:
            return True

    def _print_not_available_for(self, item):
        shared_utils.print_and_log(
            logging.WARN,
            'Conversation ID: {}, Agent ID: {} - HIT '
            'is abandoned and thus not available for '
            '{}.'.format(self.conversation_id, self.id, item),
            should_print=True,
        )

    def approve_work(self):
        """
        Approving work after it has been submitted.
        """
        if self.hit_is_abandoned:
            self._print_not_available_for('review')
        else:
            if self.hit_is_complete:
                self.m_approve_work(assignment_id=self.assignment_id)
                shared_utils.print_and_log(
                    logging.INFO,
                    'Conversation ID: {}, Agent ID: {} - HIT is '
                    'approved.'.format(self.conversation_id, self.id),
                )
            else:
                shared_utils.print_and_log(
                    logging.WARN,
                    'Cannot approve HIT. Turker hasn\'t completed the HIT yet.',
                )

    def reject_work(self, reason='unspecified'):
        """
        Reject work after it has been submitted.
        """
        if self.hit_is_abandoned:
            self._print_not_available_for('review')
        else:
            if self.hit_is_complete:
                self.m_reject_work(self.assignment_id, reason)
                shared_utils.print_and_log(
                    logging.INFO,
                    'Conversation ID: {}, Agent ID: {} - HIT is '
                    'rejected.'.format(self.conversation_id, self.id),
                )
            else:
                shared_utils.print_and_log(
                    logging.WARN,
                    'Cannot reject HIT. Turker hasn\'t completed the HIT yet.',
                )

    def block_worker(self, reason='unspecified'):
        """
        Block a worker from our tasks.
        """
        self.m_block_worker(worker_id=self.worker_id, reason=reason)
        shared_utils.print_and_log(
            logging.WARN,
            'Blocked worker ID: {}. Reason: {}'.format(self.worker_id, reason),
            should_print=True,
        )

    def pay_bonus(self, bonus_amount, reason='unspecified'):
        """
        Pays the given agent the given bonus.
        """
        if self.hit_is_abandoned:
            self._print_not_available_for('bonus')
        else:
            if self.hit_is_complete:
                unique_request_token = str(uuid.uuid4())
                self.m_pay_bonus(
                    worker_id=self.worker_id,
                    bonus_amount=bonus_amount,
                    assignment_id=self.assignment_id,
                    reason=reason,
                    unique_request_token=unique_request_token,
                )
            else:
                shared_utils.print_and_log(
                    logging.WARN,
                    'Cannot pay bonus for HIT. Reason: Turker '
                    'hasn\'t completed the HIT yet.',
                )

    def email_worker(self, subject, message_text):
        """
        Sends an email to a worker, returns true on a successful send.
        """
        response = self.m_email_worker(
            worker_id=self.worker_id, subject=subject, message_text=message_text
        )
        if 'success' in response:
            shared_utils.print_and_log(
                logging.INFO,
                'Email sent to worker ID: {}: Subject: {}: Text: {}'.format(
                    self.worker_id, subject, message_text
                ),
            )
            return True
        elif 'failure' in response:
            shared_utils.print_and_log(
                logging.WARN,
                "Unable to send email to worker ID: {}. Error: {}".format(
                    self.worker_id, response['failure']
                ),
            )
            return False

    def soft_block_worker(self, qual='block_qualification'):
        """
        Assigns this worker a soft blocking qualification.
        """
        self.m_soft_block_worker(self.worker_id, qual)

    # TODO cleanup timeout now that it's not used.
    def wait_for_hit_completion(self, timeout=None):
        """
        Waits for a hit to be marked as complete.
        """
        WAIT_TIME = 45 * 60
        start_time = time.time()
        while not self.hit_is_complete:
            if time.time() - start_time > WAIT_TIME:
                # Okay we've waited for 45 mins and the HIT still isn't up
                self.disconnected = True
            # Check if the Turker already returned/disconnected
            if self.hit_is_returned or self.disconnected:
                self.m_free_workers([self])
                return False
            # FIXME if hit_is_complete was a threading.Event() this
            # function would be cleaner and not have sleeps
            time.sleep(shared_utils.THREAD_MEDIUM_SLEEP)

        shared_utils.print_and_log(
            logging.INFO,
            'Conversation ID: {}, Agent ID: {} - HIT is done.'.format(
                self.conversation_id, self.id
            ),
        )
        self.m_free_workers([self])
        return True

    def shutdown(self, timeout=None, direct_submit=False):
        """
        Shuts down a hit when it is completed.
        """
        # Timeout in seconds, after which the HIT will be expired automatically
        # TODO clean this up, a lot can be handled by the manager instead
        # TODO it could make sense that rather than keeping the other cleanup
        # methods, agents clean up after themselves when shutdown
        if not (
            self.hit_is_abandoned
            or self.hit_is_returned
            or self.disconnected
            or self.hit_is_expired
        ):
            self.m_mark_workers_done([self])
            if direct_submit:
                self.m_send_command(
                    self.worker_id,
                    self.assignment_id,
                    {'text': data_model.COMMAND_SUBMIT_HIT},
                )
            did_complete = self.wait_for_hit_completion(timeout=timeout)
            if did_complete and self.db_logger is not None:
                self.db_logger.log_submit_assignment(self.worker_id, self.assignment_id)
            # Grab feedback message if it happens to exist
            messages = self.flush_msg_queue()
            for m in messages:
                if m['text'] == '[PEER_REVIEW]':
                    self.feedback = m['task_data']
            return did_complete

    def update_agent_id(self, agent_id):
        """
        Workaround used to force an update to an agent_id on the front-end to render the
        correct react components for onboarding and waiting worlds.

        Only really used in special circumstances where different agents need different
        onboarding worlds.
        """
        update_packet = {'agent_id': agent_id}
        self.m_send_state_change(self.worker_id, self.assignment_id, update_packet)

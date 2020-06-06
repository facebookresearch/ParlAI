#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
import uuid
import os

# Sleep constants
from parlai.mturk.core import data_model as data_model

THREAD_SHORT_SLEEP = 0.1
THREAD_MEDIUM_SLEEP = 0.3
# ThrottlingException might happen if we poll too frequently
THREAD_MTURK_POLLING_SLEEP = 10

logger = None
logging_enabled = True
debug = False
log_level = logging.ERROR

if logging_enabled:
    logging.basicConfig(
        filename=str(time.time()) + '.log',
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG,
    )
    logger = logging.getLogger('mturk')


def set_log_level(new_level):
    global log_level
    log_level = new_level


def set_is_debug(is_debug):
    global debug
    debug = is_debug


def disable_logging():
    global logging_enabled
    logging_enabled = False


def print_and_log(level, message, should_print=False):
    if (logging_enabled and level >= log_level) or debug:
        logger.log(level, message)
    if should_print or debug:  # Always print message in debug mode
        print(message)


def generate_event_id(worker_id):
    """
    Return a unique id to use for identifying a packet for a worker.
    """
    return '{}_{}'.format(worker_id, uuid.uuid4())


def get_mturk_dir():
    import parlai.mturk

    return os.path.dirname(os.path.abspath(parlai.mturk.__file__))


def get_core_dir():
    import parlai.mturk.core

    return os.path.dirname(os.path.abspath(parlai.mturk.core.__file__))


def get_tmp_dir():
    """
    Return the location of the temporary directory in which we store things related to a
    run but that can be safely deleted.
    """
    tmp_dir = os.path.join(get_mturk_dir(), 'tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    return tmp_dir


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
    STATUS_PARTNER_DISCONNECT_EARLY = 'partner disconnect early'
    STATUS_EXPIRED = 'expired'
    STATUS_RETURNED = 'returned'

    def __init__(self, status=None):
        """
        Create an AssignState to track the state of an agent's assignment.
        """
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
        """
        Appends a message to the list of messages, ensures that it is not a duplicate
        message.
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
            or self.status == self.STATUS_PARTNER_DISCONNECT_EARLY
            or self.status == self.STATUS_RETURNED
            or self.status == self.STATUS_EXPIRED
        )

    def get_inactive_command_text(self):
        """
        Get appropriate inactive command and text to respond to a reconnect given the
        current assignment state.

        returns text, command
        """
        command = data_model.COMMAND_INACTIVE_HIT
        text = None
        if self.status == self.STATUS_DISCONNECT:
            text = (
                'You disconnected in the middle of this HIT and were '
                'marked as inactive. As these HITs often require real-'
                'time interaction, it is no longer available for '
                'completion. Please return this HIT and accept a new one '
                'if you would like to try again.'
            )
        elif self.status == self.STATUS_DONE:
            command = data_model.COMMAND_INACTIVE_DONE
            text = (
                'You disconnected after completing this HIT without '
                'marking it as completed. Please press the done button '
                'below to finish the HIT.'
            )
        elif self.status == self.STATUS_EXPIRED:
            text = (
                'You disconnected in the middle of this HIT and the '
                'HIT expired before you reconnected. It is no longer '
                'available for completion. Please return this HIT and '
                'accept a new one if you would like to try again.'
            )
        elif self.status == self.STATUS_PARTNER_DISCONNECT:
            command = data_model.COMMAND_INACTIVE_DONE
            text = (
                'One of your partners disconnected in the middle of the '
                'HIT. We won\'t penalize you for their disconnect, so '
                'please use the button below to mark the HIT as complete.'
            )
        elif self.status == self.STATUS_PARTNER_DISCONNECT_EARLY:
            command = data_model.COMMAND_INACTIVE_HIT
            text = (
                'One of your partners disconnected in the middle of the '
                'HIT. We won\'t penalize you for their disconnect, but you'
                ' did not complete enough of the task to submit the HIT. '
                'Please return this HIT and accept a new one if you would '
                'like to try again.'
            )
        elif self.status == self.STATUS_RETURNED:
            text = (
                'You disconnected from this HIT and then returned '
                'it. As we have marked the HIT as returned, it is no '
                'longer available for completion. Please accept a new '
                'HIT if you would like to try again'
            )
        else:
            # We shouldn't be getting an inactive command for the other
            # states so consider this a server error
            text = (
                'Our server was unable to handle your reconnect properly '
                'and thus this HIT no longer seems available for '
                'completion. Please try to connect again or return this '
                'HIT and accept a new one.'
            )

        return text, command

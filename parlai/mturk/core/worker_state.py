# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import parlai.mturk.core.data_model as data_model

class AssignState():
    """Class for holding state information about an assignment currently
    claimed by an agent
    """

    # Possible Assignment Status Values
    STATUS_NONE = 'none'
    STATUS_ONBOARDING = 'onboarding'
    STATUS_WAITING = 'waiting'
    STATUS_ASSIGNED = 'assigned'
    STATUS_IN_TASK = 'in task'
    STATUS_DONE = 'done'
    STATUS_DISCONNECT = 'disconnect'
    STATUS_PARTNER_DISCONNECT = 'partner disconnect'
    STATUS_EXPIRED = 'expired'
    STATUS_RETURNED = 'returned'

    def __init__(self, status=None, conversation_id=None):
        """Create an AssignState with the given assignment_id. status and
        conversation_id are optional
        """
        if status == None:
            status = self.STATUS_NONE
        self.status = status
        self.messages = []
        self.last_command = None

    def clear_messages(self):
        self.messages = []
        self.last_command = None

    def is_final(self):
        """Return True if the assignment is in a final status that
        can no longer be acted on.
        """
        return (self.status == self.STATUS_DISCONNECT or
                self.status == self.STATUS_DONE or
                self.status == self.STATUS_PARTNER_DISCONNECT or
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


class WorkerState():
    """Class for holding state information about an mturk worker"""
    def __init__(self, worker_id, disconnects=0):
        """Create a new worker state for the given worker_id. Number of
        prior disconnects is optional.
        """
        self.worker_id = worker_id
        self.agents = {}
        self.disconnects = disconnects

    def active_conversation_count(self):
        """Return the number of conversations within this worker state
        that aren't in a final state
        """
        count = 0
        for assign_id in self.agents:
            if not self.agents[assign_id].state.is_final():
                count += 1
        return count

    def add_agent(self, assign_id, mturk_agent):
        """Add an assignment to this worker state with the given assign_it"""
        self.agents[assign_id] = mturk_agent

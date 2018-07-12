# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import parlai.mturk.core.data_model as data_model

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
            if not self.agents[assign_id].is_final():
                count += 1
        return count

    def add_agent(self, assign_id, mturk_agent):
        """Add an assignment to this worker state with the given assign_it"""
        self.agents[assign_id] = mturk_agent

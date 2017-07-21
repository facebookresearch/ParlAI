# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.worlds import MultiAgentDialogWorld
from joblib import Parallel, delayed

class MTurkMultiAgentDialogWorld(MultiAgentDialogWorld):
    """Basic world where each agent gets a turn in a round-robin fashion,
    receiving as input the actions of all other agents since that agent last
    acted.
    """
    def shutdown(self):
        """Shutdown all mturk agents in parallel, otherwise if one mturk agent
        is disconnected then it could prevent other mturk agents from completing."""
        global shutdown_agent
        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()
        Parallel(n_jobs=len(self.agents), backend='threading')(delayed(shutdown_agent)(agent) for agent in self.agents)

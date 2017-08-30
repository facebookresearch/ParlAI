# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.worlds import World, validate

class MTurkOnboardWorld(World):
    """Generic world for onboarding a Turker and collecting
    information from them."""
    def __init__(self, opt, mturk_agent):
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        pass

class MTurkTaskWorld(World):
    """Generic world for MTurk tasks."""
    def __init__(self, opt, mturk_agent):
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        self.mturk_agent.shutdown()
        """
        Use the following code if there are multiple MTurk agents:

        global shutdown_agent
        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()
        Parallel(
            n_jobs=len(self.mturk_agents),
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in self.mturk_agents)
        """

    def review_work(self):
        """Programmatically approve/reject the turker's work.
        For example:
        .. code-block:: python
            if self.turker_response == '0':
                self.mturk_agent.reject_work(
                    'You rated our model's response as a 0/10 but we '
                    'know we\'re better than that'
                )
            else:
                if self.turker_response == '10':
                    self.mturk_agent.pay_bonus(1, 'Thanks for a great rating!')
                self.mturk_agent.approve_work()
        """
        # self.mturk_agent.approve_work()
        # self.mturk_agent.reject_work()
        # self.mturk_agent.pay_bonus(1000) # Pay $1000 as bonus
        # self.mturk_agent.block_worker() # Block this worker from future HITs
        pass

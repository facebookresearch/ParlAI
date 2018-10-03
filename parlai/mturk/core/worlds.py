#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.worlds import World


class MTurkOnboardWorld(World):
    """Generic world for onboarding a Turker and collecting
    information from them."""
    def __init__(self, opt, mturk_agent):
        '''Init should set up resources for running the onboarding world'''
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        '''A parley should represent one turn of your onboarding task'''
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        '''Clear up resources needed for this world'''
        pass


class MTurkTaskWorld(World):
    """Generic world for MTurk tasks."""
    def __init__(self, opt, mturk_agent):
        '''Init should set up resources for running the task world'''
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        '''A parley should represent one turn of your task'''
        self.episodeDone = True

    def episode_done(self):
        '''A ParlAI-MTurk task ends and allows workers to be marked complete
        when the world is finished.
        '''
        return self.episodeDone

    def shutdown(self):
        """
        Should be used to free the world's resources and shut down the agents

        Use the following code if there are multiple MTurk agents:

        global shutdown_agent
        def shutdown_agent(mturk_agent):
            mturk_agent.shutdown()
        Parallel(
            n_jobs=len(self.mturk_agents),
            backend='threading'
        )(delayed(shutdown_agent)(agent) for agent in self.mturk_agents)
        """
        self.mturk_agent.shutdown()

    def review_work(self):
        """Programmatically approve/reject the turker's work. Doing this now
        (if possible) means that you don't need to do the work of reviewing
        later on.

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

    def save_data(self):
        '''This function should take the contents of whatever was collected
        during this task that needs to be stored for review and write it
        to disk'''
        pass

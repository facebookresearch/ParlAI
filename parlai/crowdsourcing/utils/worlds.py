#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.worlds import World


class CrowdDataWorld(World):
    def prep_save_data(self, workers):
        """
        This prepares data to be saved for later review, including chats from individual
        worker perspectives.
        """
        custom_data = self.get_custom_task_data()
        save_data = {'custom_data': custom_data, 'worker_data': {}}
        return save_data

    def get_custom_task_data(self):
        """
        This function should take the contents of whatever was collected during this
        task that should be saved and return it in some format, preferably a dict
        containing acts.

        If you need some extraordinary data storage that this doesn't cover, you can
        extend the ParlAIChatBlueprint and write your own ParlAIChatAgentState that
        defines the behavior you want.
        """
        # return {
        #     'acts': [self.important_turn1, self.important_turn2]
        #     'context': self.some_context_data_of_importance
        # }
        pass


class CrowdOnboardWorld(CrowdDataWorld):
    """
    Generic world for onboarding a Turker and collecting information from them.
    """

    def __init__(self, opt, agent):
        """
        Init should set up resources for running the onboarding world.
        """
        self.agent = agent
        self.episodeDone = False

    def parley(self):
        """
        A parley should represent one turn of your onboarding task.
        """
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Clear up resources needed for this world.
        """
        pass


class CrowdTaskWorld(CrowdDataWorld):
    """
    Generic world for Crowd tasks.
    """

    def __init__(self, opt, agent):
        """
        Init should set up resources for running the task world.
        """
        self.agent = agent
        self.episodeDone = False

    def parley(self):
        """
        A parley should represent one turn of your task.
        """
        self.episodeDone = True

    def episode_done(self):
        """
        A ParlAI-Mephisto task ends and allows workers to be marked complete when the
        world is finished.
        """
        return self.episodeDone

    def shutdown(self):
        """
        Should be used to free the world's resources and shut down the agents.
        """
        self.agent.shutdown()

    def review_work(self):
        """
        Programmatically approve/reject this work. Doing this now (if possible) means
        that you don't need to do the work of reviewing later on.

        For example:
        .. code-block:: python
            mephisto_agent = self.agent.mephisto_agent
            if self.response == '0':
                mephisto_agent.reject_work(
                    'You rated our model's response as a 0/10 but we '
                    'know we\'re better than that'
                )
            else:
                if self.response == '10':
                    mephisto_agent.pay_bonus(1, 'Thanks for a great rating!')
                mephisto_agent.approve_work()
        """
        # mephisto_agent = self.agent.mephisto_agent
        # mephisto_agent.approve_work()
        # mephisto_agent.reject_work()
        # mephisto_agent.pay_bonus(1000) # Pay $1000 as bonus
        # mephisto_agent.block_worker() # Block this worker from future work
        pass

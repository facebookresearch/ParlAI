#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.worlds import World


class MTurkDataWorld(World):
    def prep_save_data(self, workers):
        """This prepares data to be saved for later review, including
        chats from individual worker perspectives."""
        custom_data = self.get_custom_task_data()
        save_data = {
            'custom_data': custom_data,
            'worker_data': {}
        }

        for agent in workers:
            messages = agent.get_messages()
            # filter out peer feedback
            save_messages = [m for m in messages
                             if m.get('text') != '[PEER_REVIEW]']
            save_data['worker_data'][agent.worker_id] = {
                'worker_id': agent.worker_id,
                'agent_id': agent.id,
                'assignment_id': agent.assignment_id,
                'messages': save_messages,
                'given_feedback': agent.feedback,
            }

        # In simple pairing case, attach the feedback right here
        if len(workers) == 2:
            data = save_data['worker_data']
            a_0 = workers[0]
            a_1 = workers[1]
            data[a_0.worker_id]['received_feedback'] = a_1.feedback
            data[a_1.worker_id]['received_feedback'] = a_0.feedback

        return save_data

    def get_custom_task_data(self):
        """This function should take the contents of whatever was collected
        during this task that should be saved and return it in some format,
        preferrably a dict containing acts. If data needs pickling, put it
        in a field named 'needs-pickle'"""
        # return {
        #     'acts': [self.important_turn1, self.important_turn2]
        #     'context': self.some_context_data_of_importance
        #     'needs-pickle': self.json_incompatible_object
        # }
        pass


class MTurkOnboardWorld(MTurkDataWorld):
    """Generic world for onboarding a Turker and collecting
    information from them."""
    def __init__(self, opt, mturk_agent):
        """Init should set up resources for running the onboarding world"""
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        """A parley should represent one turn of your onboarding task"""
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """Clear up resources needed for this world"""
        pass


class MTurkTaskWorld(MTurkDataWorld):
    """Generic world for MTurk tasks."""
    def __init__(self, opt, mturk_agent):
        """Init should set up resources for running the task world"""
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        """A parley should represent one turn of your task"""
        self.episodeDone = True

    def episode_done(self):
        """A ParlAI-MTurk task ends and allows workers to be marked complete
        when the world is finished.
        """
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

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.worlds import validate, create_task
from parlai.mturk.core.worlds import MTurkTaskWorld, MTurkOnboardWorld


class ModelEvaluatorOnboardWorld(MTurkOnboardWorld):
    def parley(self):
        ad = {}
        ad['id'] = 'System'
        ad['text'] = 'Welcome onboard! Enter anything to confirm you\'re here.'
        self.mturk_agent.observe(ad)
        self.mturk_agent.act()
        self.episodeDone = True


class ModelEvaluatorWorld(MTurkTaskWorld):
    """
    World for letting Turkers evaluate a dialog model's performance given a context.

    Assumes the context is a context from a given task, e.g. from SQuAD, CBT, etc.
    """

    evaluator_agent_id = 'Model Evaluator'

    def __init__(self, opt, model_agent, task_opt, mturk_agent):
        self.task_world = create_task(task_opt, model_agent)
        self.mturk_agent = mturk_agent
        self.episodeDone = False

    def parley(self):
        self.task_world.parley()

        ad = {}
        # Show the dialog model's response to the context, and ask the turker
        # to rate the response
        ad['id'] = self.__class__.evaluator_agent_id
        ad['text'] = (
            self.task_world.get_acts()[0]['text']
            + "\n\n"
            + "How would you rate the following response (from 0 to 10):\n\n"
            + self.task_world.get_acts()[1]['text']
        )

        # TODO: deal with multi-turn dialogs, for now we will just deal
        # with 1-turn dialogs in this task.
        ad['episode_done'] = True  # self.world.episode_done()

        self.mturk_agent.observe(validate(ad))
        self.rating = self.mturk_agent.act()
        # Can log the rating here
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        self.task_world.shutdown()
        self.mturk_agent.shutdown()

    def review_work(self):
        pass

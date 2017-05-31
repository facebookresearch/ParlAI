# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import copy
import importlib
from parlai.core.agents import Agent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

class ModelEvaluatorAgent(Agent):
    """
    MTurk agent for evaluating a dialog model's performance given a context.
    Assumes the context is a context from a given task, e.g.
    from SQuAD, CBT, etc.
    """
    def __init__(self, opt, shared=None):
        self.opt = copy.deepcopy(opt)
        self.id = 'Model Evaluator'

        # The dialog model we will evaluate
        agent_opt = {}
        agent_opt['model'] = 'ir_baseline'
        agent = create_agent(agent_opt)

        # The task that we will evaluate the dialog model on
        task_opt = {}
        task_opt['datatype'] = 'test'
        task_opt['datapath'] = opt['datapath']
        task_opt['task'] = '#MovieDD-Reddit'
        self.world = create_task(task_opt, agent)

    def observe(self, observation):
        self.observation = observation

        # The rating given by turker
        # Because we only have one turn in this conversation, we don't need to track turn_index
        # print(self.observation)
        return observation

    def act(self):
        # All agents act once in the world
        self.world.parley()

        ad = {}
        # Show the dialog model's response to the context, and ask the turker to rate the response
        ad['text'] = (
            self.world.get_acts()[0]['text'] + "\n\n" +
            "How would you rate the following response (from 0 to 10):\n\n" +
            self.world.get_acts()[1]['text'])

        # TODO: deal with multi-turn dialogs, for now we will just deal
        # with 1-turn dialogs in this task.
        ad['episode_done'] = True  # self.world.episode_done()
        return ad

default_agent_class = ModelEvaluatorAgent

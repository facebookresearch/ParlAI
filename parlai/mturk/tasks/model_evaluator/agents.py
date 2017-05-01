# Copyright 2004-present Facebook. All Rights Reserved.

import copy
import importlib
from parlai.core.agents import Agent
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

class ModelEvaluatorAgent(Agent):
    """
    MTurk agent for evaluating bots performance given a context.
    Assumes the context is a context from a given task, e.g.
    from SQuAD, CBT, etc.
    """
    def __init__(self, opt, shared=None):
        self.opt = copy.deepcopy(opt)
        self.id = 'Model Evaluator'
        # The bot we will evaluate.
        agent_opt = {}
        agent_opt['model'] = 'ir_baseline'
        agent = create_agent(agent_opt)
        # The task that we will be collecting evaluation over.
        task_opt = {}
        task_opt['datatype'] = 'test'
        task_opt['datapath'] = opt['datapath']
        task_opt['task'] = '#MovieDD-Reddit'
        self.world = create_task(task_opt, agent)

    def act(self):
        self.world.parley()
        ad = {}
        ad['text'] = (
            self.world.query['text'] + "\n\n" +
            "How would you rate the following response (from 0 to 10):\n\n" +
            self.world.reply['text'])
        # TODO: deal with multi-turn dialogs, for now we will just deal
        # with 1-turn dialogs in this task.
        ad['episode_done'] = True  # self.world.episode_done()
        return ad

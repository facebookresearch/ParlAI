# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import copy
import importlib
from parlai.core.agents import Agent
from parlai.core.agents import create_agent

class QADataCollectionAgent(Agent):
    """
    MTurk agent for recording a turker's question and answer given a context.
    Assumes the context is a random context from a given task, e.g.
    from SQuAD, CBT, etc.
    """
    def __init__(self, opt, shared=None):
        self.opt = copy.deepcopy(opt)
        self.id = 'QA Collector'
        self.turn_index = -1

        # Initialize a SQuAD teacher agent, which we will later get context from
        module_name = 'parlai.tasks.squad.agents'
        class_name = 'DefaultTeacher'
        my_module = importlib.import_module(module_name)
        task_class = getattr(my_module, class_name)
        task_opt = {}
        task_opt['datatype'] = 'train'
        task_opt['datapath'] = opt['datapath']
        self.task = task_class(task_opt)

    def observe(self, observation):
        self.observation = observation

        if self.turn_index == 0:
            # Turker's question, from the first turn
            # print(self.observation)
            pass
        elif self.turn_index == 1:
            # Turker's answer, from the second turn
            # print(self.observation)
            pass
        return observation

    def act(self):
        self.turn_index = (self.turn_index + 1) % 2; # Each turn starts from the QA Collector agent
        ad = { 'episode_done': False }
        ad['id'] = self.id

        if self.turn_index == 0:
            # At the first turn, the QA Collector agent provides the context and
            # prompts the turker to ask a question regarding the context

            # Get context from SQuAD teacher agent
            qa = self.task.act()
            context = '\n'.join(qa['text'].split('\n')[:-1])

            # Wrap the context with a prompt telling the turker what to do next
            ad['text'] = (context +
                        '\n\nPlease provide a question given this context.')

        if self.turn_index == 1:
            # At the second turn, the QA Collector collects the turker's question from the first turn,
            # and then prompts the turker to provide the answer

            # A prompt telling the turker what to do next
            ad['text'] = 'Thanks. And what is the answer to your question?'

            ad['episode_done'] = True  # end of episode

        return ad

default_agent_class = QADataCollectionAgent

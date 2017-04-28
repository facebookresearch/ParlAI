# Copyright 2004-present Facebook. All Rights Reserved.

import copy
import importlib
from parlai.core.agents import Agent
from parlai.core.agents import create_agent

class QADataCollectionAgent(Agent):
    """
    MTurk agent for recording questions and answers given a context.
    Assumes the context is a random context from a given task, e.g.
    from SQuAD, CBT, etc.
    """
    def __init__(self, opt, shared=None):
        self.opt = copy.deepcopy(opt)
        self.id = 'QA Collector'
        self.turn_index = -1
        
        module_name = 'parlai.tasks.squad.agents'
        class_name = 'DefaultTeacher'
        my_module = importlib.import_module(module_name)
        task_class = getattr(my_module, class_name)
        task_opt = {}
        task_opt['datatype'] = 'train'
        task_opt['datapath'] = opt['datapath']
        self.task = task_class(task_opt)

    def act(self):
        self.turn_index = (self.turn_index + 1) % 3;
        ad = { 'episode_done': False }
        ad['id'] = self.id
        if self.turn_index == 0:
            # get context
            qa = self.task.act()
            context = '\n'.join(qa['text'].split('\n')[:-1])
            ad['text'] = (context + 
                        '\n\nPlease provide a question given this context.')
        if self.turn_index == 1:
            ad['text'] = 'Thanks. And what is the answer to your question?'
        if self.turn_index == 2:
            ad['text'] = 'Thanks again!'
            ad['episode_done'] = True  # end of episode
        return ad

# Copyright 2004-present Facebook. All Rights Reserved.

import copy
from parlai.core.agents import Agent

class QADataCollectionAgent(Agent):
    """
    MTurk agent for recording questions and answers given a context.
    Assumes the context is a random context from a given task, e.g.
    from SQuAD, CBT, etc.
    """
    def __init__(self, opt, shared=None):
        self.opt = copy.deepcopy(opt)
        self.id = 'qa_collector'
        self.conversation_id = None
        self.turn_index = -1

    def act(self):
        self.turn_index = (self.turn_index + 1) % 3;
        ad = { 'episode_done': False }
        ad['id'] = self.id
        if self.turn_index == 0:
            ad['text'] = ('Context' + 
                        '\nPlease provide a question given this context.')
        if self.turn_index == 1:
            ad['text'] = 'Thanks. And what is the answer to your question?'
        if self.turn_index == 2:
            ad['text'] = 'Thanks again!'
            ad['episode_done'] = True  # end of episode
        return ad

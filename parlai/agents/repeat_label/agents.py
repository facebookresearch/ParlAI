#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import random

from parlai.core.agents import Agent


class RepeatLabelAgent(Agent):

    def __init__(self, opt, shared=None):
        self.returnSingleRandomAnswer = True
        self.cantAnswerPercent = 0
        super().__init__(opt)

    def act(self, obs):
        reply = {}
        if obs.get('labels', False):
            labels = obs['labels']
            if random.random() >= self.cantAnswerPercent:
                if self.returnSingleRandomAnswer:
                    reply['text'] = labels[random.randrange(len(labels))]
                else:
                    reply['text'] = ', '.join(labels)
            else:
                # Some 'self.cantAnswerPercent' percentage of the time
                # the agent does not answer.
                reply['text'] = "I don't know."
        return reply

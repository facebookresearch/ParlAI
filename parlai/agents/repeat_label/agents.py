#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import random

from parlai.core.agents import Agent


class RepeatLabelAgent(Agent):

    def __init__(self, opt):
        self.returnSingleRandomAnswer = True
        super().__init__(opt)

    def act(self, obs):
        reply = {}
        if obs.get('labels', False):
            labels = obs['labels']
            if self.returnSingleRandomAnswer:
                reply['text'] = labels[random.randrange(len(labels))]
            else:
                reply['text'] = ', '.join(labels)
        return reply

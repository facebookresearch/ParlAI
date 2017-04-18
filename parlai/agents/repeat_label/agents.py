#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import random

from parlai.core.agents import Agent


class RepeatLabelAgent(Agent):

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.returnSingleRandomAnswer = True
        self.cantAnswerPercent = 0
        self.id = 'RepeatLabelAgent'


    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()
        if 'labels' in obs and len(obs['labels']) > 0:
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
        else:
            reply['text'] = "I don't know."

        if 'label_candidates' in obs and len(obs['label_candidates']) > 0:
            # Produce text_candidates by randomly ordering all other
            # candidate labels.
            cands = [ reply['text'] ]
            y = list(obs['label_candidates'])
            random.shuffle(y)
            for k in y:
                if k != reply['text']:
                    cands.append(k)
            reply['text_candidates'] = cands
        return reply

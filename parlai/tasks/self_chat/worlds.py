#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

from parlai.core.worlds import DialogPartnerWorld, validate


class InteractiveWorld(DialogPartnerWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.init_contexts()
        self.max_cnt = self.opt.get('selfchat_max_turns', 10)
        self.cnt = 0

    def init_contexts(self):
        pass

    def get_contexts(self):
        return ['__SILENCE__', '']

    def display(self):
        s = ''
        s += super().display()
        if self.cnt == 0:
            s += '\n==============================\n'
        return s

    def episode_done(self):
        if self.cnt > self.max_cnt:
            return True
        else:
            return False

    def parley(self):
        if self.episode_done():
            self.cnt = 0
            agents = self.get_agents()
            for a in agents:
                a.reset()

        if self.cnt == 0:
            self.acts = [None, None]
            # choose speaking order:
            if random.choice([0, 1]):
                self.agents_ordered = [self.agents[0], self.agents[1]]
            else:
                self.agents_ordered = [self.agents[1], self.agents[0]]
            self.contexts = self.get_contexts()
            # initial context
            for i in range(0, 2):
                context = {
                    'text': self.contexts[i],
                    'episode_done': False,
                    'id': 'context',
                }
                self.acts[1 - i] = context
                self.agents_ordered[i].observe(validate(context))
        else:
            # do regular loop
            acts = self.acts
            agents = self.agents_ordered
            acts[0] = agents[0].act()
            agents[1].observe(validate(acts[0]))
            acts[1] = agents[1].act()
            agents[0].observe(validate(acts[1]))

        self.update_counters()
        self.cnt += 1

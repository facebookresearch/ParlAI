#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Simple agent which repeats back the last thing said to it.
Useful as a baseline for metrics like F1.
"""

from parlai.core.agents import Agent


class RepeatQueryAgent(Agent):

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RepeatQueryAgent'

    def act(self):
        obs = self.observation
        if obs is None:
            return {'text': 'Nothing to repeat yet.'}
        reply = {}
        reply['id'] = self.getID()
        query = obs.get('text', "I don't know")
        # Take last line if there are multiple lines.
        reply['text'] = query.split('\n')[-1]
        return reply

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Simple agent which repeats back the last thing said to it.
Useful as a baseline for metrics like F1.
"""

import random

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

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Simple agent which repeats back the last thing said to it.

Useful as a baseline for metrics like F1.
"""

from parlai.core.agents import Agent
from parlai.core.message import Message


class RepeatQueryAgent(Agent):
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RepeatQueryAgent'

    def act(self):
        obs = self.observation
        if obs is None:
            return Message({'text': 'Nothing to repeat yet.', 'episode_done': False})
        reply = {}
        reply['id'] = self.getID()
        query = obs.get('text', "I don't know")
        # Take last line if there are multiple lines.
        reply['text'] = query.split('\n')[-1]
        if reply['text'] == '':
            reply['text'] = 'Nothing to repeat yet.'
        reply['episode_done'] = False
        return Message(reply)

    def batch_act(self, observations):
        batch_reply = []
        original_obs = self.observation
        for obs in observations:
            self.observation = obs
            batch_reply.append(self.act())
        self.observation = original_obs
        return batch_reply

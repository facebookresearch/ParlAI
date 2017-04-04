# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.core.agents import Agent


class RepeatLabelAgent(Agent):

    def act(self, obs):
        reply = {}
        if obs.get('labels', False):
            reply['text'] = ', '.join(obs['labels'])
        return reply

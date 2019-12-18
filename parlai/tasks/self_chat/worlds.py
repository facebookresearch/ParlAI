#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict, List

from parlai.core.agents import Agent
from parlai.core.worlds import DialogPartnerWorld, validate


class SelfChatBaseWorld(DialogPartnerWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.init_contexts()
        self.init_openers()
        self.max_turn_cnt = self.opt.get('selfchat_max_turns', 10)
        self.turn_cnt = 0
        self.episode_cnt = 0

    def init_contexts(self):
        pass

    def get_contexts(self, episode_num: int) -> List[str]:
        return ['__SILENCE__', '']

    def init_openers(self) -> None:
        pass

    def get_openers(self, episode_num: int) -> List[str]:
        pass

    def display(self):
        s = ''
        s += super().display()
        if self.turn_cnt == 0:
            s += '\n==============================\n'
        return s

    def episode_done(self):
        return self.turn_cnt >= self.max_turn_cnt

    def _get_seed_utt_acts(
        self, episode_num: int, agents: List[Agent]
    ) -> List[Dict[str, Any]]:
        def make_agent_action(utterance: str, agent: Agent) -> Dict[str, Any]:
            return {'text': utterance, 'episode_done': False, 'id': agent.id}

        openers = self.get_openers(episode_num)
        if not openers:
            return []
        return list(map(make_agent_action, openers, agents))

    def parley(self):
        if self.episode_done():
            self.turn_cnt = 0
            self.episode_cnt += 1
            self.contexts = None
            self.seed_utterances = None
            agents = self.get_agents()
            for a in agents:
                a.reset()

        if self.turn_cnt == 0:
            self.acts = [None, None]
            # choose speaking order:
            if random.choice([0, 1]):
                self.agents_ordered = [self.agents[0], self.agents[1]]
            else:
                self.agents_ordered = [self.agents[1], self.agents[0]]
            # get the beginning of the conversation
            self.contexts = self.get_contexts(self.episode_cnt)
            self.seed_utterances = self._get_seed_utt_acts(
                self.episode_cnt, self.agents_ordered
            )

        if self.contexts:
            # initial context
            for i in range(0, 2):
                context = {
                    'text': self.contexts[i],
                    'episode_done': False,
                    'id': 'context',
                }
                self.acts[1 - i] = context
                self.agents_ordered[i].observe(validate(context))
            self.contexts = None
        elif self.seed_utterances:
            # grab the first two seed messages
            utts = self.seed_utterances[:2]
            # process the turn
            for i in [0, 1]:
                # if we have a seed utterance, add it to the conversation
                if len(utts) > i:
                    self.acts[i] = utts[i]
                    if hasattr(self.agents_ordered[i], 'self_observe'):
                        self.agents_ordered[i].self_observe(self.acts[i])
                else:
                    self.acts[i] = self.agents_ordered[i].act()
                self.agents_ordered[1 - i].observe(validate(self.acts[i]))
            # remove the used seed messages from the queue
            self.seed_utterances = self.seed_utterances[2:]
        else:
            # do regular loop
            acts = self.acts
            agents = self.agents_ordered
            acts[0] = agents[0].act()
            agents[1].observe(validate(acts[0]))
            acts[1] = agents[1].act()
            agents[0].observe(validate(acts[1]))

        self.update_counters()
        self.turn_cnt += 1


class InteractiveWorld(SelfChatBaseWorld):
    pass

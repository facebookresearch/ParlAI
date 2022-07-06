#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
from typing import Any, Dict, List, Optional

from parlai.agents.fixed_response.fixed_response import FixedResponseAgent
from parlai.core.agents import Agent
from parlai.core.worlds import create_task, DialogPartnerWorld, validate
from parlai.core.message import Message


def load_openers(opt) -> Optional[List[str]]:
    if opt['task'].startswith('internal:') or opt['task'].startswith('fb:'):
        base_task = opt['task']
    else:
        base_task = opt['task'].split(':')[0]

    if base_task == 'self_chat':
        # TODO(#2284): Load default openers from s3
        return None

    print('[ loading conversation openers... ]')
    # create dummy task so we can get openers from the data
    task_opt = copy.deepcopy(opt)
    task_opt['task'] = base_task

    # default train will loop forever, but evalmode will stop after one epoch
    datatype = task_opt['datatype']
    if 'train' in datatype and 'evalmode' not in datatype:
        task_opt['datatype'] = f'{datatype}:evalmode'
    task_opt['interactive_task'] = False
    task_opt['selfchat_task'] = False
    task_opt['fixed_response'] = None
    task_agent = FixedResponseAgent(task_opt)
    task_world = create_task(task_opt, task_agent)

    # run through task data, collecting all first messages
    openers = []
    is_first_turn = True
    while not task_world.epoch_done():
        task_world.parley()
        msg = task_world.get_acts()[0]
        # add only the first message in the episode
        if is_first_turn and msg.get('text'):
            openers.append(msg['text'])
        is_first_turn = msg.get('episode_done', False)

    # remove duplicates while preserving the ordering of the loaded openers
    openers = list(dict.fromkeys(openers))

    print(f'[ loaded {len(openers)} openers ]')
    return openers


def load_openers_from_file(filepath: str) -> List[str]:
    openers = []
    with open(filepath, 'r') as f:
        openers = [l.strip() for l in f]
    return openers


class SelfChatWorld(DialogPartnerWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.init_contexts(shared=shared)
        self._openers = None
        self.init_openers()
        self.max_turn_cnt = self.opt.get('selfchat_max_turns', 10)
        self.turn_cnt = 0
        self.episode_cnt = 0

    def init_contexts(self, shared=None) -> None:
        """
        Override to load or instantiate contexts to be used to seed the self chat.
        """
        pass

    def get_contexts(self):
        """
        Override to return a pair of contexts with which to seed the self chat episode.

        This function will be called before the first turn of every episode.
        """
        return ['Hi!', '']

    def init_openers(self) -> None:
        """
        Override to load or instantiate opening messages to be used to seed the self
        chat.
        """
        if self.opt.get('seed_messages_from_task'):
            self._openers = load_openers(self.opt)
        elif self.opt.get('seed_messages_from_file'):
            self._openers = load_openers_from_file(self.opt['seed_messages_from_file'])

    def get_openers(self, episode_num: int) -> Optional[List[str]]:
        """
        Override to return one or more opening messages with which to seed the self chat
        episode.

        The return value should be an array of strings, each string being a message in
        response to the string before it.
        """
        if self._openers:
            return [random.choice(self._openers)]
        return None

    def display(self):
        s = super().display()
        if self.turn_cnt == 0:
            s += '\n==============================\n'
        return s

    def episode_done(self):
        return self.turn_cnt >= self.max_turn_cnt

    def _get_seed_utt_acts(
        self, episode_num: int, agents: List[Agent]
    ) -> List[Dict[str, Any]]:
        """
        Return acts of any utterances to "seed" the conversation with.
        """

        def make_agent_action(utterance: str, agent: Agent) -> Dict[str, Any]:
            return {'text': utterance, 'episode_done': False, 'id': agent.id}

        if self.turn_cnt == 0:
            # Create the seed utterances from any openers
            openers = self.get_openers(episode_num)
            if not openers:
                return []
            return list(map(make_agent_action, openers, agents))
        else:
            # Just return the existing seed utterances, if any exist
            return self.seed_utterances

    def parley(self):
        if self.episode_done():
            self._end_episode()

        if self.turn_cnt == 0:
            self.acts = [None, None]
            # get any context for the beginning of the conversation
            self.contexts = self.get_contexts()

        self.seed_utterances = self._get_seed_utt_acts(self.episode_cnt, self.agents)

        if self.contexts:
            assert len(self.contexts) == 2
            # initial context
            for i in range(0, 2):
                context = Message(
                    {'text': self.contexts[i], 'episode_done': False, 'id': 'context'}
                )
                self.acts[i] = context
                self.agents[i].observe(validate(context))
            # clear contexts so they are only added once per episode
            self.contexts = None
        elif self.seed_utterances:
            # pop the next two seed messages (there may be less or more than 2 total)
            utts = self.seed_utterances[:2]
            self.seed_utterances = self.seed_utterances[2:]
            # process the turn
            for i in [0, 1]:
                # if we have a seed utterance, add it to the conversation
                if len(utts) > i:
                    self.acts[i] = utts[i]
                    if hasattr(self.agents[i], 'self_observe'):
                        self.agents[i].observe({'episode_done': False})
                        self.agents[i].self_observe(self.acts[i])
                else:
                    self.acts[i] = self.agents[i].act()
                self.agents[1 - i].observe(validate(self.acts[i]))
        else:
            # do regular loop
            acts = self.acts
            agents = self.agents
            acts[0] = agents[0].act()
            agents[1].observe(validate(acts[0]))
            acts[1] = agents[1].act()
            agents[0].observe(validate(acts[1]))

        self.update_counters()
        self.turn_cnt += 1

    def _end_episode(self):
        """
        Apply logic to end the episode.
        """
        self.turn_cnt = 0
        self.episode_cnt += 1
        self.contexts = None
        self.seed_utterances = None
        self.reset_agents()

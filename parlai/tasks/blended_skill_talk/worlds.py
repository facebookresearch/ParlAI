#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import os
from copy import deepcopy

from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld


def load_personas(opt):
    print('[ loading personas.. ]')
    fname = os.path.join(
        opt['datapath'], 'blended_skill_talk', opt['datatype'] + '.json'
    )
    with open(fname) as json_file:
        data = json.load(json_file)
    contexts = []
    for d in data:
        p1 = (
            'your persona: '
            + d['personas'][0][0]
            + '\n'
            + 'your persona: '
            + d['personas'][0][1]
            + '\n'
        )
        p2 = (
            'your persona: '
            + d['personas'][1][0]
            + '\n'
            + 'your persona: '
            + d['personas'][1][1]
            + '\n'
        )
        if d['context_dataset'] == 'wizard_of_wikipedia':
            p1 += d['additional_context'] + '\n'
            p2 += d['additional_context'] + '\n'
        ctxt = d['free_turker_utterance'] + '\n' + d['guided_turker_utterance']
        p1 += ctxt
        p2 += ctxt
        contexts.append([p1, p2])
    return contexts


class InteractiveWorld(DialogPartnerWorld):
    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('BST Interactive World')
        parser.add_argument(
            '--display-partner-persona',
            type='bool',
            default=True,
            help='Display your partner persona at the end of the chat',
        )

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.contexts = load_personas(self.opt)
        self.display_partner_persona = self.opt['display_partner_persona']
        self.cnt = 0

    def get_new_personas(self):
        random.seed()
        p = random.choice(self.contexts)
        return p[0], p[1]

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        if self.cnt == 0:
            self.p1, self.p2 = self.get_new_personas()

        acts = self.acts
        human_agent, model_agent = self.agents
        if self.cnt == 0:
            # add the persona on to the first message to human agent
            act = {}
            act['text'] = self.p1
            act['episode_done'] = False
            act['id'] = 'persona'
            human_agent.observe(validate(act))
        act = deepcopy(human_agent.act())
        if self.cnt == 0:
            # add the persona on to the first message to model agent
            act.force_set('text', self.p2 + '\n' + act.get('text', 'hi'))
            model_agent.observe(validate(act))
        else:
            model_agent.observe(validate(act))
        acts[1] = model_agent.act()
        human_agent.observe(validate(acts[1]))
        self.update_counters()
        self.cnt += 1

        if act['episode_done']:
            print("\nCHAT DONE.\n")
            if self.display_partner_persona:
                partner_persona = self.p2.replace(
                    'your persona:', 'partner\'s persona:'
                )
                print(
                    f"Your partner was playing the following persona:\n{partner_persona}"
                )
            print("[ Preparing new chat ... ]\n")
            self.cnt = 0


class SelfChatWorld(SelfChatBaseWorld):
    def init_contexts(self):
        self.contexts_data = load_personas(self.opt)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return [p[0], p[1]]

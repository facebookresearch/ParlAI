#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.worlds import create_task
from parlai.agents.fixed_response.fixed_response import FixedResponseAgent
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld

import random


def get_personas(opt, shared=None):
    if shared and 'personas_list' in shared:
        return shared['personas_list']
    return _load_personas(opt=opt)


def _load_personas(opt):
    print('[ loading personas.. ]')
    # Create ConvAI2 data so we can assign personas.
    convai2_opt = opt.copy()
    convai2_opt['task'] = 'convai2:both'
    if convai2_opt['datatype'].startswith('train'):
        convai2_opt['datatype'] = 'train:evalmode'
    convai2_opt['interactive_task'] = False
    convai2_opt['selfchat_task'] = False
    convai2_agent = FixedResponseAgent({'fixed_response': None})
    convai2_world = create_task(convai2_opt, convai2_agent)
    personas = set()
    while not convai2_world.epoch_done():
        convai2_world.parley()
        msg = convai2_world.get_acts()[0]
        # Find a new episode
        if msg.get('episode_done', False) and not convai2_world.epoch_done():
            convai2_world.parley()
            msg = convai2_world.get_acts()[0]
            txt = msg.get('text', '').split('\n')
            a1_persona = []
            a2_persona = []
            for t in txt:
                if t.startswith("partner's persona:"):
                    a1_persona.append(t.replace("partner's persona:", 'your persona:'))
                if t.startswith('your persona:'):
                    a2_persona.append(t)
            personas.add('\n'.join(a1_persona))
            personas.add('\n'.join(a2_persona))
    print('[ loaded ' + str(len(personas)) + ' personas ]')
    return list(personas)


class InteractiveWorld(InteractiveBaseWorld):
    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('ConvAI2 Interactive World')
        parser.add_argument(
            '--display-partner-persona',
            type='bool',
            default=True,
            help='Display your partner persona at the end of the chat',
        )

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.display_partner_persona = self.opt['display_partner_persona']

    def init_contexts(self, shared=None):
        self.personas_list = get_personas(opt=self.opt, shared=shared)

    def get_contexts(self):
        random.seed()
        personas_1 = random.choice(self.personas_list)
        personas_2 = random.choice(self.personas_list)
        return personas_1, personas_2

    def finalize_episode(self):
        print("\nCHAT DONE.\n")
        if self.display_partner_persona:
            partner_persona = self.p2.replace('your persona:', 'partner\'s persona:')
            print(f"Your partner was playing the following persona:\n{partner_persona}")
        if not self.epoch_done():
            print("[ Preparing new chat ... ]\n")

    def share(self):
        shared_data = super().share()
        shared_data['personas_list'] = self.personas_list
        return shared_data


class SelfChatWorld(SelfChatBaseWorld):
    def init_contexts(self, shared=None):
        self.personas_list = get_personas(self.opt, shared=shared)

    def get_contexts(self):
        random.seed()
        personas_1 = random.choice(self.personas_list)
        personas_2 = random.choice(self.personas_list)
        return [personas_1, personas_2]

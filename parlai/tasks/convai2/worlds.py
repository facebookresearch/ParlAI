#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from parlai.core.worlds import create_task
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.tasks.self_chat.worlds import SelfChatBaseWorld

import random
from typing import List


def load_personas(opt):
    print('[ loading personas.. ]')
    # Create ConvAI2 data so we can assign personas.
    convai2_opt = opt.copy()
    convai2_opt['task'] = 'convai2:both'
    if convai2_opt['datatype'].startswith('train'):
        convai2_opt['datatype'] = 'train:evalmode'
    convai2_opt['interactive_task'] = False
    convai2_agent = RepeatLabelAgent(convai2_opt)
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
            a1_persona = ""
            a2_persona = ""
            for t in txt:
                if t.startswith("partner's persona:"):
                    a1_persona += (
                        t.replace("partner's persona:", 'your persona:') + '\n'
                    )
                if t.startswith('your persona:'):
                    a2_persona += t + '\n'
            personas.add(a1_persona)
            personas.add(a2_persona)
    print('[ loaded ' + str(len(personas)) + ' personas ]')
    return list(personas)


class InteractiveWorld(DialogPartnerWorld):
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
        self.personas_list = load_personas(self.opt)
        self.display_partner_persona = self.opt['display_partner_persona']
        self.cnt = 0

    def get_new_personas(self):
        random.seed()
        personas_1 = random.choice(self.personas_list)
        personas_2 = random.choice(self.personas_list)
        return personas_1, personas_2

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
            act.force_set('text', self.p2 + act.get('text', 'hi'))
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


class InteractiveSelfchatWorld(SelfChatBaseWorld):
    def init_contexts(self):
        self.personas_list = load_personas(self.opt)

    def get_contexts(self, episode_num: int) -> List[str]:
        random.seed()
        personas_1 = random.choice(self.personas_list)
        personas_2 = random.choice(self.personas_list)
        return [personas_1, personas_2]

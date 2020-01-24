#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.worlds import create_task
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.tasks.self_chat.worlds import InteractiveWorld as SelfChatBaseWorld
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent

from copy import deepcopy
import random
import pickle
import os


class InteractiveSimpleWorld(DialogPartnerWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print("[ loading personas.. ]")
        self.load_personas()

    def load_personas(self):
        # Create Light data so we can assign personas.
        light_opt = self.opt.copy()
        light_opt['task'] = 'light_dialog'
        light_opt['interactive_task'] = False
        light_agent = RepeatLabelAgent(light_opt)
        self.light_world = create_task(light_opt, light_agent)
        self.cnt = 0

    def get_new_personas(self):
        # Find a new episode
        while True:
            self.light_world.parley()
            msg = self.light_world.get_acts()[0]
            if msg.get('episode_done', False):
                self.light_world.parley()
                msg = self.light_world.get_acts()[0]
                break
        txt = msg.get('text', '').split('\n')
        a1_persona = ""  # (typically human in interactive)
        a2_persona = ""
        p = {}
        for t in txt:
            p[t.split(' ')[0]] = t

        a1_persona = (
            ' _task_speech\n'
            + p['_setting_name']
            + '\n'
            + p['_setting_desc']
            + '\n'
            + p['_self_name'].replace("_self_name", '_partner_name')
            + '\n'
            + p['_partner_name'].replace("_partner_name", '_self_name')
            + '\n'
            + '_self_persona I am a '
            + ' '.join(p['_partner_name'].split(' ')[1:])
            + '.\n'
        )

        a2_persona = (
            ' _task_speech\n'
            + p['_setting_name']
            + '\n'
            + p['_setting_desc']
            + '\n'
            + p['_partner_name']
            + '\n'
            + p['_self_name']
            + '\n'
            + p['_self_persona']
            + '\n'
        )
        return a1_persona, a2_persona

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        if self.cnt == 0:
            self.p1, self.p2 = self.get_new_personas()

        acts = self.acts
        agents = self.agents
        if self.cnt == 0:
            # add the persona on to the first message to agent 0
            act = {}
            act['text'] = self.p1
            act['episode_done'] = False
            act['id'] = 'persona'
            agents[0].observe(validate(act))
        act = deepcopy(agents[0].act())
        acts[0] = act
        if self.cnt == 0:
            # add the persona on to the first message to agent 1
            act.force_set('text', self.p2 + act.get('text', 'hi'))
            agents[1].observe(validate(act))
        else:
            agents[1].observe(validate(act))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()
        self.cnt += 1

        if act['episode_done']:
            print("CHAT DONE ")
            print("\n... preparing new chat... \n")
            self.cnt = 0


class InteractiveSelfchatWorld(SelfChatBaseWorld):
    def init_contexts(self):
        print('[ loading contexts.. ]')
        data_path = os.path.join(
            self.opt['datapath'], 'light_dialogue', 'light_environment.pkl'
        )
        file = open(data_path, 'rb')
        self.db = pickle.load(file)
        # compact list of rooms
        rs = []
        for _k, r in self.db['rooms'].items():
            rs.append(r)
        self.db['rooms'] = rs
        # compact list of characters
        cs = []
        for _k, c in self.db['characters'].items():
            cs.append(c)
        self.db['all_characters'] = cs

    def make_context(self, room, c1, c2):
        s = '_task_speech\n'
        s += (
            '_setting_name '
            + room.get('setting', '')
            + ', '
            + room.get('category', '')
            + '\n'
        )
        s += '_setting_desc ' + room.get('description', '') + '\n'
        s += '_partner_name ' + c2.get('name', '') + '\n'
        s += '_self_name ' + c1.get('name', '') + '\n'
        s += '_self_persona ' + random.choice(c1.get('personas', ['']))
        return s

    def get_contexts(self):
        room = random.choice(self.db['rooms'])
        if len(room.get('in_characters', [])) > 0:
            c1 = self.db['characters'][random.choice(room['in_characters'])]
        else:
            c1 = random.choice(self.db['all_characters'])
        c2 = random.choice(self.db['all_characters'])
        p1 = self.make_context(room, c1, c2)
        p2 = self.make_context(room, c2, c1)
        return [p1, p2]

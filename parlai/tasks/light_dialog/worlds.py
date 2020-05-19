#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.worlds import create_task
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent

import random
import pickle
import os


class InteractiveSimpleWorld(InteractiveBaseWorld):
    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('LIGHT Interactive World')
        parser.add_argument(
            '--add-task-string',
            type='bool',
            default=False,
            help='Add _task_speech to text input to model or not',
        )

    def init_contexts(self, shared=None):
        # Create Light data so we can assign personas.
        light_opt = self.opt.copy()
        light_opt['task'] = 'light_dialog'
        light_opt['interactive_task'] = False
        light_agent = RepeatLabelAgent(light_opt)
        self.light_world = create_task(light_opt, light_agent)
        self.cnt = 0

    def get_contexts(self):
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

        if self.opt['add_task_string']:
            task_name = ' _task_speech\n'

        else:
            task_name = ''

        a1_persona = (
            task_name
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
            + '.'
        )

        a2_persona = (
            task_name
            + p['_setting_name']
            + '\n'
            + p['_setting_desc']
            + '\n'
            + p['_partner_name']
            + '\n'
            + p['_self_name']
            + '\n'
            + p['_self_persona']
        )
        return a1_persona, a2_persona


class SelfChatWorld(SelfChatBaseWorld):
    def init_contexts(self, shared=None):
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

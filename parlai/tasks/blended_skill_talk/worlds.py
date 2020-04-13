#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import os

from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld


def load_personas(opt):
    print('[ loading personas.. ]')
    fname = os.path.join(
        opt['datapath'], 'blended_skill_talk', opt['datatype'].split(':')[0] + '.json'
    )
    with open(fname) as json_file:
        data = json.load(json_file)
    contexts = []
    for d in data:
        if opt.get('include_personas', True):
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
        else:
            p1 = ''
            p2 = ''
        if d['context_dataset'] == 'wizard_of_wikipedia':
            p1 += d['additional_context'] + '\n'
            p2 += d['additional_context'] + '\n'
        ctxt = d['free_turker_utterance'] + '\n' + d['guided_turker_utterance']
        p1 += ctxt
        p2 += ctxt
        contexts.append([p1, p2])
    return contexts


class InteractiveWorld(InteractiveBaseWorld):
    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('BST Interactive World')
        parser.add_argument(
            '--display-partner-persona',
            type='bool',
            default=True,
            help='Display your partner persona at the end of the chat',
        )
        parser.add_argument(
            '--include-personas',
            type='bool',
            default=True,
            help='Include personas as input context, or not',
        )

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.display_partner_persona = self.opt['display_partner_persona']

    def init_contexts(self, shared=None):
        self.contexts_data = load_personas(self.opt)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return p[0], p[1]

    def finalize_episode(self):
        print("\nCHAT DONE.\n")
        if self.display_partner_persona:
            partner_persona = self.p2.replace('your persona:', 'partner\'s persona:')
            print(f"Your partner was playing the following persona:\n{partner_persona}")
        print("[ Preparing new chat ... ]\n")


class SelfChatWorld(SelfChatBaseWorld):
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('BST SelfChat World')
        parser.add_argument(
            '--include-personas',
            type='bool',
            default=True,
            help='Include personas as input context, or not',
        )

    def init_contexts(self, shared=None):
        self.contexts_data = load_personas(self.opt)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return [p[0], p[1]]

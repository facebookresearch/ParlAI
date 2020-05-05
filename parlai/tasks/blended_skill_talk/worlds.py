#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import random

from parlai.tasks.blended_skill_talk.agents import raw_data_path
from parlai.tasks.interactive.worlds import InteractiveWorld as InteractiveBaseWorld
from parlai.tasks.self_chat.worlds import SelfChatWorld as SelfChatBaseWorld


def get_contexts_data(opt, shared=None):
    if shared and 'contexts_data' in shared:
        return shared['contexts_data']
    return _load_personas(opt=opt)


def _load_personas(opt):
    print('[ loading personas.. ]')
    if opt.get('include_personas', True):
        print(
            "\n  [NOTE: In the BST paper both partners have a persona.\n"
            + '         You can choose to ignore yours, the model never sees it.\n'
            + '         In the Blender paper, this was not used for humans.\n'
            + '         You can also turn personas off with --include-personas False]\n'
        )
    fname = raw_data_path(opt)
    with open(fname) as json_file:
        data = json.load(json_file)
    contexts = []
    for d in data:
        context1 = []
        context2 = []
        if opt.get('include_personas', True):
            context1.append('your persona: ' + d['personas'][0][0])
            context1.append('your persona: ' + d['personas'][0][1])
            context2.append('your persona: ' + d['personas'][1][0])
            context2.append('your persona: ' + d['personas'][1][1])
        if d['context_dataset'] == 'wizard_of_wikipedia':
            context1.append(d['additional_context'])
            context2.append(d['additional_context'])
        if opt.get('include_initial_utterances', True):
            context1.append(d['free_turker_utterance'])
            context2.append(d['free_turker_utterance'])
            context1.append(d['guided_turker_utterance'])
            context2.append(d['guided_turker_utterance'])
        c1 = '\n'.join(context1)
        c2 = '\n'.join(context2)
        contexts.append([c1, c2])
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
        parser.add_argument(
            '--include-initial-utterances',
            type='bool',
            default=False,
            help='Include context conversation at beginning or not',
        )

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.display_partner_persona = self.opt['display_partner_persona']

    def init_contexts(self, shared=None):
        self.contexts_data = get_contexts_data(self.opt, shared=shared)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return p[0], p[1]

    def finalize_episode(self):
        print("\nCHAT DONE.\n")
        if self.display_partner_persona:
            partner_persona = self.p2.replace('your persona:', 'partner\'s persona:')
            print(f"Your partner was playing the following persona:\n{partner_persona}")
        if not self.epoch_done():
            print("\n[ Preparing new chat ... ]\n")

    def share(self):
        shared_data = super().share()
        shared_data['contexts_data'] = self.contexts_data
        return shared_data


class SelfChatWorld(SelfChatBaseWorld):
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('BST SelfChat World')
        parser.add_argument(
            '--include-personas',
            type='bool',
            default=True,
            help='Include personas as input context, or not',
        )
        parser.add_argument(
            '--include-initial-utterances',
            type='bool',
            default=True,
            help='Include context conversation at beginning or not',
        )

    def init_contexts(self, shared=None):
        self.contexts_data = get_contexts_data(self.opt, shared=shared)

    def get_contexts(self):
        random.seed()
        p = random.choice(self.contexts_data)
        return [p[0], p[1]]

    def share(self):
        shared_data = super().share()
        shared_data['contexts_data'] = self.contexts_data
        return shared_data

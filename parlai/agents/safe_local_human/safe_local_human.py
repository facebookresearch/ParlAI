#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent that gets the local keyboard input in the act() function.

Applies safety classifier(s) to process user and partner messages.
"""

from parlai.core.message import Message
from parlai.utils.misc import display_messages
from parlai.utils.strings import colorize
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier


class SafeLocalHumanAgent(LocalHumanAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Safe Local Human Arguments')
        agent.add_argument(
            '--safety',
            type=str,
            default='all',
            choices={'none', 'string_matcher', 'classifier', 'all'},
            help='Apply safety filtering to messages',
        )
        super(SafeLocalHumanAgent, cls).add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'safeLocalHuman'
        self.init_safety(opt)

    def init_safety(self, opt):
        if opt['safety'] == 'string_matcher' or opt['safety'] == 'all':
            self.offensive_string_matcher = OffensiveStringMatcher()
        if opt['safety'] == 'classifier' or opt['safety'] == 'all':
            self.offensive_classifier = OffensiveLanguageClassifier()
        self.self_offensive = False

    def offensive(self, text):
        if (
            hasattr(self, 'offensive_string_matcher')
            and text in self.offensive_string_matcher
        ):
            return True
        if hasattr(self, 'offensive_classifier') and text in self.offensive_classifier:
            return True
        return False

    def observe(self, msg):
        if not self.self_offensive:
            # check offensiveness of other agent.
            if not self.offensive(msg.get('text', '')):
                print(
                    display_messages(
                        [msg],
                        ignore_fields=self.opt.get('display_ignore_fields', ''),
                        prettify=self.opt.get('display_prettify', False),
                    )
                )
            else:
                # do not print anything at all.
                pass

    def act(self):
        reply = Message()
        reply['id'] = self.getID()
        reply_text = input(colorize("Enter Your Message:", 'field') + ' ')
        reply_text = reply_text.replace('\\n', '\n')
        if self.offensive(reply_text):
            print("[ Sorry, could not process that message. ]")
            self.self_offensive = True
        else:
            self.self_offensive = False
        if self.opt.get('single_turn', False):
            reply_text += '[DONE]'
        reply['episode_done'] = False
        reply['label_candidates'] = self.fixedCands_txt
        if '[DONE]' in reply_text:
            reply.force_set('episode_done', True)
            self.episodeDone = True
            reply_text = reply_text.replace('[DONE]', '')
        reply['text'] = reply_text
        if '[EXIT]' in reply_text:
            self.finished = True
        return reply

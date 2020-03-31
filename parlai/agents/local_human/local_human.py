#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent does gets the local keyboard input in the act() function.

Example: python examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid
"""

from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.utils.misc import display_messages, load_cands
from parlai.utils.strings import colorize
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier


class LocalHumanAgent(Agent):
    def add_cmdline_args(argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        agent = argparser.add_argument_group('Local Human Arguments')
        agent.add_argument(
            '-fixedCands',
            '--local-human-candidates-file',
            default=None,
            type=str,
            help='File of label_candidates to send to other agent',
        )
        agent.add_argument(
            '--single_turn',
            type='bool',
            default=False,
            help='If on, assumes single turn episodes.',
        )
        agent.add_argument(
            '--safety',
            type=str,
            default='none',
            choices={'none', 'string_matcher', 'classifier', 'all'},
            help='Apply safety filtering to messages',
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'localHuman'
        self.episodeDone = False
        self.finished = False
        self.fixedCands_txt = load_cands(self.opt.get('local_human_candidates_file'))
        self.init_safety(opt)
        print(
            colorize(
                "Enter [DONE] if you want to end the episode, [EXIT] to quit.",
                'highlight',
            )
        )

    def epoch_done(self):
        return self.finished

    def init_safety(self, opt):
        if opt['safety'] == 'string_matcher' or opt['safety'] == 'all':
            self.offensive_string_matcher = OffensiveStringMatcher()
        if opt['safety'] == 'classifier' or opt['safety'] == 'all':
            self.offensive_classifier = OffensiveLanguageClassifier()

    def offensive(self, text):
        if hasattr(
            self, 'offensive_string_matcher'
        ) and self.offensive_string_matcher.__contains__(text):
            return True
        if hasattr(
            self, 'offensive_classifier'
        ) and self.offensive_classifier.__contains__(text):
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

    def episode_done(self):
        return self.episodeDone

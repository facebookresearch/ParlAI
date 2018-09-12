# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Simple agent which chooses a random label from the label candidates if
they are available. If candidates are not available, it repeats the label.
"""

import random

from parlai.core.agents import Agent


class RandomCandidateAgent(Agent):

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('RandomCandidateAgent Arguments')
        parser.add_argument(
            '--label_candidates_file', type=str, default=None,
            help='file of candidate responses to choose from')

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'RandomCandidateAgent'
        random.seed(42)
        if opt.get('label_candidates_file'):
            f = open(opt.get('label_candidates_file'))
            self.label_candidates = f.read().split('\n')
            
    def act(self):
        obs = self.observation
        if obs is None:
            return {'text': 'Nothing to reply to yet.'}
        reply = {}
        reply['id'] = self.getID()
        label_candidates = obs.get('label_candidates')
        if hasattr(self, 'label_candidates'):
            label_candidates = self.label_candidates
        if label_candidates:
            random.shuffle(label_candidates)
            reply['text_candidates'] = label_candidates
            reply['text'] = label_candidates[0]
        else:
            # reply with I don't know.
            reply['text'] = "I don't know."

        return reply

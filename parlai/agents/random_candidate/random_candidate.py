#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Simple agent which chooses a random label.

Chooses from the label candidates if they are available.
If candidates are not available, it repeats the label.
"""

import random

from parlai.core.agents import Agent


class RandomCandidateAgent(Agent):
    """Agent returns random candidate if available or repeats the label."""

    @staticmethod
    def add_cmdline_args(parser):
        """Add command line arguments for this agent."""
        parser = parser.add_argument_group('RandomCandidateAgent Arguments')
        parser.add_argument(
            '--label_candidates_file', type=str, default=None,
            help='file of candidate responses to choose from')

    def __init__(self, opt, shared=None):
        """Initialize this agent."""
        super().__init__(opt)
        self.id = 'RandomCandidateAgent'
        random.seed(42)
        if opt.get('label_candidates_file'):
            f = open(opt.get('label_candidates_file'))
            self.label_candidates = f.read().split('\n')

    def act(self):
        """Generate response to last seen observation.

        Replies with a randomly selected candidate if label_candidates or a
        candidate file are available.
        Otherwise, replies with the label if they are available.
        Oterhwise, replies with generic hardcoded responses if the agent has
        not observed any messages or if there are no replies to suggest.

        :returns: message dict with reply
        """
        obs = self.observation
        if obs is None:
            return {'text': 'Nothing to reply to yet.'}
        reply = {}
        reply['id'] = self.getID()
        label_candidates = obs.get('label_candidates')
        if hasattr(self, 'label_candidates'):
            # override label candidates with candidate file if set
            label_candidates = self.label_candidates
        if label_candidates:
            label_candidates = list(label_candidates)
            random.shuffle(label_candidates)
            reply['text_candidates'] = label_candidates
            reply['text'] = label_candidates[0]
        else:
            # reply with I don't know.
            reply['text'] = "I don't know."

        return reply

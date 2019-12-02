#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent uses the Free Alice AIML interpreter to answer..
"""

import aiml
import os

from parlai.core.agents import Agent


class AliceAgent(Agent):
    """
    Agent returns random candidate if available or repeats the label.
    This is a strong rule-based baseline.
    """

    @staticmethod
    def add_cmdline_args(parser):
        """
        Add command line arguments for this agent.
        """
        parser = parser.add_argument_group('AliceAgent Arguments')
        parser.add_argument(
            '--label_candidates_file',
            type=str,
            default=None,
            help='file of candidate responses to choose from',
        )

    def __init__(self, opt, shared=None):
        """
        Initialize this agent.
        """
        super().__init__(opt)
        self.id = 'Alice'
        if opt.get('label_candidates_file'):
            f = open(opt.get('label_candidates_file'))
            self.label_candidates = f.read().split('\n')
        self.kern = None
        self.load_alice()

    def load_alice(self):
        self.kern = aiml.Kernel()
        self.kern.setTextEncoding(None)
        chdir = os.path.join( aiml.__path__[0],'botdata','alice' )
        self.kern.bootstrap(learnFiles="startup.xml", commands="load alice",
                            chdir=chdir)
    
    def get_alice_response(self):
        return self.kern.respond(self.observation['text'])

    def act(self):
        """
        Generate response to last seen observation.

        Replies with a message from using the Alice bot.

        :returns: message dict with reply
        """
        if self.observation is None:
            return {'text': 'Nothing to reply to yet.'}
        
        reply = {}
        reply['id'] = self.getID()
        reply['text'] = self.get_alice_response()

        label_candidates = None
        if hasattr(self, 'label_candidates'):
            # override label candidates with candidate file if set
            label_candidates = self.label_candidates
            
        if label_candidates:
            label_candidates = list(label_candidates)
            reply['text_candidates'] = label_candidates

        return reply

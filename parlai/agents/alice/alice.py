#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent uses the Free ALICE AIML interpreter to generate replies to observations.

More information can be found at:
https://en.wikipedia.org/wiki/Artificial_Linguistic_Internet_Computer_Entity
"""

import os
from parlai.core.agents import Agent

try:
    import aiml

    IMPORT_OKAY = True
except ImportError:
    IMPORT_OKAY = False


class AliceAgent(Agent):
    """
    Agent returns the Alice AIML bot's reply to an observation.

    This is a strong rule-based baseline.
    """

    def __init__(self, opt, shared=None):
        """
        Initialize this agent.
        """
        if not IMPORT_OKAY:
            raise ImportError(
                "ALICE agent needs python-aiml installed. Please run:\n "
                "`pip install git+https://github.com/paulovn/python-aiml.git`."
            )

        super().__init__(opt)
        self.id = 'Alice'
        self.kern = None
        if shared is None:
            self._load_alice()
        else:
            self.kern = shared['kern']

    def share(self):
        shared = super().share()
        shared['kern'] = self.kern
        return shared

    def _load_alice(self):
        self.kern = aiml.Kernel()
        self.kern.verbose(False)
        self.kern.setTextEncoding(None)
        chdir = os.path.join(aiml.__path__[0], 'botdata', 'alice')
        self.kern.bootstrap(
            learnFiles="startup.xml", commands="load alice", chdir=chdir
        )

    def act(self):
        """
        Generate response to last seen observation.

        Replies with a message from using the Alice bot.

        :returns: message dict with reply
        """

        obs = self.observation
        if obs is None:
            return {'text': 'Nothing to reply to yet.'}

        reply = {}
        reply['id'] = self.getID()
        query = obs.get('text', "I don't know")
        reply['text'] = self.kern.respond(query)

        return reply

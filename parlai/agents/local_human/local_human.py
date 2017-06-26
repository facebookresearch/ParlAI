# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Agent does gets the local keyboard input in the act() function.
   Example: python examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid
"""

from parlai.core.agents import Agent
from parlai.core.worlds import display_messages

class LocalHumanAgent(Agent):

    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'localHuman'
        self.episodeDone = False

    def observe(self, msg):
        print(display_messages([msg]))

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()
        reply_text = input("Enter Your Message: ")
        reply_text = reply_text.replace('\\n', '\n')
        reply['episode_done'] = False
        if '[DONE]' in reply_text:
            reply['episode_done'] = True
            self.episodeDone = True
            reply_text = reply_text.replace('[DONE]', '')
        reply['text'] = reply_text
        return reply

    def episode_done(self):
        return self.episodeDone

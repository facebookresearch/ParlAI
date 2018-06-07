# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Agent does gets the local keyboard input in the act() function.
   Example: python examples/eval_model.py -m local_human -t babi:Task1k:1 -dt valid
"""

from parlai.core.agents import Agent
from parlai.core.utils import display_messages, load_cands

class LocalHumanAgent(Agent):

    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Local Human Arguments')
        agent.add_argument('-fixedCands', '--local-human-candidates-file',
                           default=None, type=str,
                           help='File of label_candidates to send to other agent')
    
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.id = 'localHuman'
        self.episodeDone = False
        self.fixedCands_txt = load_cands(self.opt.get('local_human_candidates_file'))
            
    def observe(self, msg):
        print(display_messages([msg],
                               ignore_fields=self.opt.get('display_ignore_fields', ''),
                               prettify=self.opt.get('display_prettify', False)))

    def act(self):
        obs = self.observation
        reply = {}
        reply['id'] = self.getID()
        reply_text = input("Enter Your Message: ")
        reply_text = reply_text.replace('\\n', '\n')
        reply['episode_done'] = False
        reply['label_candidates'] = self.fixedCands_txt
        if '[DONE]' in reply_text:
            reply['episode_done'] = True
            self.episodeDone = True
            reply_text = reply_text.replace('[DONE]', '')
        reply['text'] = reply_text
        return reply

    def episode_done(self):
        return self.episodeDone

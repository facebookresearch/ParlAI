#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.core.message import Message

import logging

class InteractiveWorld(DialogPartnerWorld):
    """
    Simple interactive world involving just two agents talking.

    In more sophisticated worlds the environment could supply information, e.g. in
    tasks/convai2 both agents are given personas, so a world class should be written
    especially for those cases for given tasks.
    """

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.init_contexts(shared=shared)
        self.turn_cnt = 0
        self.first_time = True

    def init_contexts(self, shared=None):
        """
        Override to load or instantiate contexts to be used to seed the chat.
        """
        pass

    def get_contexts(self):
        """
        Override to return a pair of contexts with which to seed the episode.

        This function will be called before the first turn of every episode.
        """
        return ['', '']

    def finalize_episode(self):
        print("CHAT DONE ")
        if not self.epoch_done():
            print("\n... preparing new chat... \n")

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        
        acts = self.acts
        agents = self.agents

        if self.first_time:
            agents[0].observe(
                {
                    'id': 'World',
                    'text': 'Hello!',
                }
            )
            self.first_time = False
            return
 
        try:
            act = deepcopy(agents[0].act())
        except StopIteration:
            self.reset()
            self.finalize_episode()
            self.turn_cnt = 0
            return

        if not act:
            return

        act_text = act.get('text', None)
        logging.info('Act: ' + str(act))
        message_history = act.get('message_history', [])
        logging.info(str(message_history))

        acts[0] = act
        if act_text and '[DONE]' in act_text:
            agents[0].observe(validate(Message({'text': 'Goodbye!', 'episode_done': True})))
            self.reset()
            return

        logging.info('Starting observing message history')
        for message in message_history + [act_text]:
            if message and message.startswith('your persona:'):
                act = Message({'id': 'context', 'text': message,
                               'episode_done': False})
            else:
                act = Message({'text': message, 'episode_done': False})
            logging.info(f'Observing message {message}')
            agents[1].observe(validate(act))

        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()
        self.turn_cnt += 1

        if act['episode_done']:
            self.finalize_episode()
            self.turn_cnt = 0

        self.reset()

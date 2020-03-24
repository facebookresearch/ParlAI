#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from parlai.core.worlds import DialogPartnerWorld, validate

class InteractiveWorld(DialogPartnerWorld):
    """
    Simple interactive world involving just two agents talking.

    In more sophisticated worlds the environment could supply information, e.g. in
    tasks/convai2 both agents are given personas, so a world class should be written
    especially for those cases for given tasks.
    """
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.init_contexts()
        self.turn_cnt = 0
        
    def init_contexts(self):
        """
        Override to load or instantiate contexts to be used to seed the chat.
        """
        pass

    def get_contexts(self):
        """
        Override to return a pair of contexts with which to seed the episode.

        This function will be called before the first turn of every episode.
        """
        return ['__SILENCE__', '']

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        if self.turn_cnt == 0:
            self.p1, self.p2 = self.get_contexts()

        acts = self.acts
        agents = self.agents
        if self.turn_cnt == 0:
            # add the context on to the first message to agent 0
            act = {}
            act['text'] = self.p1
            act['episode_done'] = False
            act['id'] = 'context'
            agents[0].observe(validate(act))
        act = deepcopy(agents[0].act())
        acts[0] = act
        if self.turn_cnt == 0:
            # add the context on to the first message to agent 1
            act.force_set('text', self.p2 + act.get('text', 'hi'))
            agents[1].observe(validate(act))
        else:
            agents[1].observe(validate(act))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()
        self.turn_cnt += 1

        if act['episode_done']:
            print("CHAT DONE ")
            print("\n... preparing new chat... \n")
            self.turn_cnt = 0

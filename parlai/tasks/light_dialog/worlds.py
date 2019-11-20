#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from parlai.core.worlds import create_task
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent


class InteractiveWorld(DialogPartnerWorld):
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print("[ loading personas.. ]")
        self.load_personas()

    def load_personas(self):
        # Create Light data so we can assign personas.
        light_opt = self.opt.copy()
        light_opt['task'] = 'light_dialog'
        light_opt['interactive_task'] = False
        light_agent = RepeatLabelAgent(light_opt)
        self.light_world = create_task(light_opt, light_agent)
        self.cnt = 0

    def get_new_personas(self):
        # Find a new episode
        while True:
            self.light_world.parley()
            msg = self.light_world.get_acts()[0]
            if msg.get('episode_done', False):
                self.light_world.parley()
                msg = self.light_world.get_acts()[0]
                break
        txt = msg.get('text', '').split('\n')
        a1_persona = ""  # (typically human in interactive)
        a2_persona = ""
        for t in txt:
            if not t.startswith("_partner_say"):
                if t.startswith("_partner_name"):
                    t0 = t.replace("_partner_name", '_self_name')
                else:
                    t0 = t.replace("_self_name", '_partner_name')
                if t.startswith("_self_persona"):
                    t0 = ''
                if t.startswith("_object_desc"):
                    continue
                if t.startswith("_self_act") or t.startswith("_self_emote"):
                    continue
                if t0 != '':
                    a1_persona += t0 + '\n'
                if t != '':
                    a2_persona += t + '\n'
        return a1_persona, a2_persona

    def parley(self):
        """Agent 0 goes first. Alternate between the two agents."""
        if self.cnt == 0:
            self.p1, self.p2 = self.get_new_personas()

        acts = self.acts
        agents = self.agents
        if self.cnt == 0:
            # add the persona on to the first message to agent 0
            act = {}
            act['text'] = self.p1
            act['episode_done'] = False
            act['id'] = 'persona'
            agents[0].observe(validate(act))
        act = deepcopy(agents[0].act())
        if self.cnt == 0:
            # add the persona on to the first message to agent 1
            act.force_set('text', self.p2 + act.get('text', 'hi'))
            agents[1].observe(validate(act))
        else:
            agents[1].observe(validate(act))
        acts[1] = agents[1].act()
        agents[0].observe(validate(acts[1]))
        self.update_counters()
        self.cnt += 1

        if act['episode_done']:
            print("CHAT DONE ")
            print("\n... preparing new chat... \n")
            self.cnt = 0

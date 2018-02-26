##
## Copyright (c) 2017-present, Facebook, Inc.
## All rights reserved.
## This source code is licensed under the BSD-style license found in the
## LICENSE file in the root directory of this source tree. An additional grant
## of patent rights can be found in the PATENTS file in the same directory.
##
from parlai.core.worlds import World, ExecutableWorld, validate
from projects.mastering_the_dungeon.tasks.graph_world2.graph import construct_graph

class GraphWorld2(ExecutableWorld):

    def init_world(self):
        self.g = construct_graph(self.opt)
        for a in self.agents:
            self.g.new_agent(a.id)

    def observe(self, agent, act):
        if agent.id == act['id']:
            msg = {}
            msg['text'] = self.g.get_text(agent.id).rstrip('\n')
            msg['id'] = 'world'
            msg['graph'] = self.g # preferably agents don't use the graph directly,
                                  # but we make available here.
            return msg            
        else:
            return None

    def execute(self, agent, act):
        # Execute action from agent. We also send an update to all other agents
        # that can observe the change.
        if 'text' in act:
            valid = self.g.parse_exec(agent.id, act['text'])
            if not valid:
                self.g.send_msg(agent.id, 'Invalid action.\n')
        self.g.update_world() # other NPCs can move, etc.

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.mturk.core.worlds import MTurkTaskWorld
from parlai.core.worlds import validate
from joblib import Parallel, delayed
from parlai.tasks.dealnodeal.agents import NegotiationTeacher
import random

class MTurkDealNoDealDialogWorld(MTurkTaskWorld):
    """Basic world where each agent gets a turn in a round-robin fashion,
    receiving as input the actions of all other agents since that agent last
    acted.
    """
    def __init__(self, opt, agents=None, shared=None):
        # Add passed in agents directly.
        if agents is not None:
            random.shuffle(agents)
        self.agents = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False
        self.task = NegotiationTeacher(opt=opt)
        self.first_turn = True
        self.choices = dict()
        self.selection = False
        self.turns = 0
        self.num_negotiations = 0


    def parley(self):
        """For each agent, get an observation of the last action each of the
        other agents took. Then take an action yourself.
        """
        if self.first_turn:
            self.num_negotiations += 1

            # Use NegotiationTeacher to load data for us
            self.task.dialogue_idx = None
            act = self.task.act()
            for other_agent in self.agents:
                other_agent.observe(validate(act))
            self.first_turn = False
        else:
            self.turns += 1

            for index, agent in enumerate(self.agents):
                if agent in self.choices:
                    # This agent has already made a choice
                    continue

                try:
                    act = agent.act(timeout=None)
                except TypeError:
                    act = agent.act()  # not MTurkAgent

                for other_agent in self.agents:
                    if other_agent != agent:
                        other_agent.observe(validate(act))

                if act["text"].startswith("<selection>") and self.turns > 1:
                    # Making a choice
                    self.choices[agent] = act["text"]
                    self.selection = True
                    if len(self.choices) == len(self.agents):
                        self.first_turn = True
                        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """Shutdown all mturk agents in parallel, otherwise if one mturk agent
        is disconnected then it could prevent other mturk agents from completing."""
        global shutdown_agent
        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except:
                agent.shutdown() # not MTurkAgent
        Parallel(n_jobs=len(self.agents), backend='threading')(delayed(shutdown_agent)(agent) for agent in self.agents)

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.worlds import validate
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
import time

class TestOnboardWorld(MTurkOnboardWorld):
    TEST_ID = 'ONBOARD_SYSTEM'
    TEST_TEXT_1 = 'FIRST_ONBOARD_MESSAGE'
    TEST_TEXT_2 = 'SECOND_ONBOARD_MESSAGE'

    def parley(self):
        ad = {}
        ad['id'] = self.TEST_ID
        ad['text'] = self.TEST_TEXT_1
        self.mturk_agent.observe(ad)
        response = self.mturk_agent.act()
        self.mturk_agent.observe({
            'id': self.TEST_ID,
            'text': self.TEST_TEXT_2
        })
        response = self.mturk_agent.act()
        self.episodeDone = True


class TestSoloWorld(MTurkTaskWorld):
    """
    World for taking 2 turns and then marking the worker as done
    """

    TEST_ID = 'SYSTEM'
    TEST_TEXT_1 = 'FIRST_MESSAGE'
    TEST_TEXT_2 = 'SECOND_MESSAGE'

    def __init__(self, opt, task, mturk_agent):
        self.task = task
        self.mturk_agent = mturk_agent
        self.episodeDone = False
        self.turn_index = -1

    def parley(self):
        self.turn_index = (self.turn_index + 1) % 2
        ad = { 'episode_done': False }
        ad['id'] = self.__class__.TEST_ID

        if self.turn_index == 0:
            # Take a first turn
            ad['text'] = self.TEST_TEXT_1

            self.mturk_agent.observe(validate(ad))
            self.response1 = self.mturk_agent.act()

        if self.turn_index == 1:
            # Complete after second turn
            ad['text'] = self.TEST_TEXT_2

            ad['episode_done'] = True  # end of episode

            self.mturk_agent.observe(validate(ad))
            self.response2 = self.mturk_agent.act()

            time.sleep(1)
            self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def report(self):
        pass

    def shutdown(self):
        self.mturk_agent.shutdown(timeout=-1)
        pass

    def review_work(self):
        pass

class TestDuoWorld(MTurkTaskWorld):
    """World where 2 participants send messages in a circle for 2 rounds"""

    MESSAGE_1 = 'TEST_MESSAGE_1'
    MESSAGE_2 = 'TEST_MESSAGE_2'
    MESSAGE_3 = 'TEST_MESSAGE_3'
    MESSAGE_4 = 'TEST_MESSAGE_4'

    def __init__(self, opt, agents=None, shared=None):
        # Add passed in agents directly.
        self.agents = agents
        self.acts = [None] * len(agents)
        self.episodeDone = False
        self.rounds = 0

    def parley(self):
        """For each agent, act, then force other agents to observe your act
        """
        acts = self.acts
        for index, agent in enumerate(self.agents):
            try:
                acts[index] = agent.act(timeout=None)
            except TypeError:
                acts[index] = agent.act() # not MTurkAgent
            if acts[index]['episode_done']:
                self.episodeDone = True
            for other_agent in self.agents:
                if other_agent != agent:
                    other_agent.observe(validate(acts[index]))
        self.rounds += 1
        if self.rounds >= 2:
            time.sleep(2)
            self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        for index, agent in enumerate(self.agents):
            agent.shutdown(timeout=-1)

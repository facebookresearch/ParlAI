#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# py parlai/messenger/tasks/overworld_demo/run.py --debug --verbose

from parlai.core.worlds import World


class MessengerBotChatTaskWorld(World):
    """Example one person world that talks to a provided agent (bot)."""

    MAX_AGENTS = 1

    def __init__(self, opt, agent, bot):
        self.agent = agent
        self.episodeDone = False
        self.model = bot

    @staticmethod
    def run(messenger_manager, opt, agents, task_id):
        agent = agents[0]
        world = MessengerBotChatTaskWorld(
            opt=opt,
            agent=agent
        )
        while not world.episode_done():
            world.parley()
        world.shutdown()

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'ChatbotAgent'

    def parley(self):
        a = self.agent.act()
        if a is not None:
            if '[DONE]' in a['text']:
                self.episodeDone = True
            else:
                print("===act====")
                print(a)
                print("~~~~~~~~~~~")
                self.model.observe(a)
                response = self.model.act()
                print("===response====")
                print(response)
                print("~~~~~~~~~~~")
                response['id'] = ''
                self.agent.observe(response)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()

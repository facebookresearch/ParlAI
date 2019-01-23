#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.worlds import World
from parlai.messenger.core.worlds import OnboardWorld


# ---------- Echo demo ---------- #
class MessengerEchoOnboardWorld(OnboardWorld):
    """Example messenger onboarding world for Echo task, displays.
    onboarding worlds that only exist to send an introduction message.
    """
    @staticmethod
    def run(opt, agent, task_id):
        world = MessengerEchoOnboardWorld(opt=opt, agent=agent)
        while not world.episode_done():
            world.parley()
        world.shutdown()

    def parley(self):
        self.agent.observe({
            'id': 'Onboarding',
            'text': 'Welcome to the onboarding world for our echo bot. '
                    'The next message you send will be echoed. Use [DONE] '
                    'to finish the chat.'
        })
        self.episodeDone = True


class MessengerEchoTaskWorld(World):
    """Example one person world that uses only user input."""

    MAX_AGENTS = 1

    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False

    @staticmethod
    def run(messenger_manager, opt, agents, task_id):
        agent = agents[0]
        world = MessengerEchoTaskWorld(
            opt=opt,
            agent=agent
        )
        while not world.episode_done():
            world.parley()
        world.shutdown()

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'EchoAgent'

    def parley(self):
        a = self.agent.act()
        if a is not None:
            if '[DONE]' in a['text']:
                self.episodeDone = True
            else:
                a['id'] = 'World'
                self.agent.observe(a)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()


# ----------- Onboard Data Demo ----------- #
class MessengerOnboardDataOnboardWorld(OnboardWorld):
    """Example messenger onboarding that collects and returns data for use
    in the real task world
    """
    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False
        self.turn = 0

    @staticmethod
    def run(opt, agent, task_id):
        '''Runs an instance of the onboarding world. Data returned from
        here will be made available in the agent.onboard_data field'''
        world = MessengerOnboardDataOnboardWorld(opt=opt, agent=agent)
        while not world.episode_done():
            world.parley()
        world.shutdown()
        return {'name': world.user_name, 'color': world.fav_color}

    def parley(self):
        if self.turn == 0:
            self.agent.observe({
                'id': 'Onboarding',
                'text': 'Welcome to the onboarding world the onboarding '
                        'data demo.\nEnter your name.',
            })
            a = self.agent.act()
            while a is None:
                a = self.agent.act()
            self.user_name = a['text']
            self.turn = self.turn + 1
        elif self.turn == 1:
            self.agent.observe({
                'id': 'Onboarding',
                'text': '\nEnter your favorite color.',
            })
            a = self.agent.act()
            while a is None:
                a = self.agent.act()
            self.fav_color = a['text']
            self.episodeDone = True


class MessengerOnboardDataTaskWorld(World):
    """Example one person world that relays data given in the onboard world."""

    MAX_AGENTS = 1

    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False

    @staticmethod
    def run(messenger_manager, opt, agents, task_id):
        agent = agents[0]
        world = MessengerOnboardDataTaskWorld(
            opt=opt,
            agent=agent
        )
        while not world.episode_done():
            world.parley()
        world.shutdown()

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'DataAgent'

    def parley(self):
        name = self.agent.onboard_data['name']
        color = self.agent.onboard_data['color']
        self.agent.observe({
            'id': 'World',
            'text': 'During onboarding, you said your name was {} and your '
                    'favorite color was {}'.format(name, color)
        })
        self.episodeDone = True

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()


# ---------- Chat world -------- #
class MessengerChatOnboardWorld(OnboardWorld):
    """Example messenger onboarding world for chat task, displays intro and
    explains instructions.
    """
    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False
        self.turn = 0

    @staticmethod
    def run(opt, agent, task_id):
        world = MessengerChatOnboardWorld(opt=opt, agent=agent)
        while not world.episode_done():
            world.parley()
        world.shutdown()
        return {'name': world.user_name}

    def parley(self):
        if self.turn == 0:
            self.agent.observe({
                'id': 'Onboarding',
                'text': 'Welcome to the onboarding world free chat. '
                        'Enter your display name.'
            })
            a = self.agent.act()
            while a is None:
                a = self.agent.act()
            self.user_name = a['text']
            self.turn = self.turn + 1
        elif self.turn == 1:
            self.agent.observe({
                'id': 'Onboarding',
                'text': 'You will be matched with a random person. Say [DONE] '
                        'to end the chat.',
            })
            self.episodeDone = True


class MessengerChatTaskWorld(World):
    """Example one person world that lets two users chat."""

    MAX_AGENTS = 2

    def __init__(self, opt, agents):
        self.agents = agents
        self.episodeDone = False

    @staticmethod
    def run(messenger_manager, opt, agents, task_id):
        world = MessengerChatTaskWorld(
            opt=opt,
            agents=agents
        )
        while not world.episode_done():
            world.parley()
        world.shutdown()

    @staticmethod
    def assign_roles(agents):
        for agent in agents:
            agent.disp_id = agent.onboard_data['name']

    def parley(self):
        for x in [0, 1]:
            a = self.agents[x].act()
            if a is not None:
                if '[DONE]' in a['text']:
                    self.agents[x - 1].observe({
                        'id': 'World',
                        'text': 'The other agent has ended the chat.'
                    })
                    self.episodeDone = True
                else:
                    self.agents[x - 1].observe(a)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        for agent in self.agents:
            agent.shutdown()


# ---------- Overworld -------- #
class MessengerOverworld(World):
    """World to handle moving agents to their proper places"""
    DEMOS = {
        'echo': (MessengerEchoOnboardWorld, MessengerEchoTaskWorld),
        'onboard data': (MessengerOnboardDataOnboardWorld,
                         MessengerOnboardDataTaskWorld),
        'chat': (MessengerChatOnboardWorld, MessengerChatTaskWorld),
    }

    def __init__(self, opt, agent):
        self.agent = agent
        self.opt = opt
        self.first_time = True

    def return_overworld(self):
        self.first_time = True

    def parley(self):
        if self.first_time:
            self.agent.observe({
                'id': 'Overworld',
                'text': 'Welcome to the overworld for the ParlAI messenger '
                        'demo. Choose one of the demos from the listed quick '
                        'replies. ',
                'quick_replies': self.DEMOS.keys(),
            })
            self.first_time = False
        a = self.agent.act()
        if a is not None:
            if a['text'] in self.DEMOS:
                self.agent.observe({
                    'id': 'Overworld',
                    'text': 'Transferring to ' + a['text'],
                })
                return a['text']
            else:
                self.agent.observe({
                    'id': 'Overworld',
                    'text': 'Invalid option. Choose one of the demos from the '
                            'listed quick replies. ',
                    'quick_replies': self.DEMOS.keys(),
                })

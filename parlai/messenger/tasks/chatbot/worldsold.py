# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# py parlai/messenger/tasks/overworld_demo/run.py --debug --verbose

from parlai.core.worlds import World
from parlai.messenger.core.worlds import OnboardWorld

class MessengerBotChatOnboardWorld(OnboardWorld):
    """Example messenger onboarding world for Echo task, displays.
    onboarding worlds that only exist to send an introduction message.
    """
    @staticmethod
    def run(agent, task_id):
        world = MessengerBotChatOnboardWorld(opt=None, agent=agent)
        while not world.episode_done():
            world.parley()
        world.shutdown()

    def parley(self):
        self.agent.observe({
            'id': 'Onboarding',
            'text': "Hey how's it going?"
                    "Use [DONE] to finish the chat."
        })
        self.episodeDone = True

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

class MessengerBotChatTaskWorld(World):
    """Example one person world that uses only user input."""

    MAX_AGENTS = 1

    def __init__(self, opt, agent):
        self.agent = agent
        self.episodeDone = False
        # create model
        opt = {}
        opt["datapath"]="/Users/jase/src/ParlAI/data"
        opt["model"]="/Users/jase/src/ParlAI/parlai_external.projects.personachat.memnn1hop.memnn1hop:Memnn1hopAgent"
        opt["model_file"]="/Users/jase/src/ParlAI/parlai_external/data/personachat/memnn2hop_sweep/persona-none_rephraseTrn-True_rephraseTst-False_lr-0.1_esz-500_margin-0.1_tfidf-False_shareEmb-True_hops0_lins0/model"
        opt['fixed_candidates_file'] = "/Users/jase/src/ParlAI/data/personachat/cands.txt"
        print(opt)
        self.model = create_agent(opt)

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
        agents[0].disp_id = 'EchoAgent'

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
                # response['text'] = "lol"
                self.agent.observe(response)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()



# ---------- Overworld -------- #
class MessengerOverworld(World):
    """World to handle moving agents to their proper places"""
    DEMOS = {
        'echo': (MessengerEchoOnboardWorld, MessengerEchoTaskWorld),
        'humanchat': (MessengerChatOnboardWorld, MessengerChatTaskWorld),
        'botchat': (MessengerBotChatOnboardWorld, MessengerBotChatTaskWorld),
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

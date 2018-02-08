# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# py parlai/messenger/tasks/overworld_demo/run.py --debug --verbose

from parlai.core.worlds import World
from parlai.messenger.core.worlds import OnboardWorld
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

class MessengerBotChatTaskWorld(World):
    """Example one person world that talks to a provided agent (bot)."""

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
        #----
        opt['model_file'] = None
        opt['model'] = 'repeat_label'
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

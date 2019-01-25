#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.messenger.tasks.qa_data_collection.worlds import \
    QADataCollectionWorld
from parlai.messenger.core.messenger_manager import MessengerManager
from parlai.messenger.core.worlds import SimpleMessengerOverworld as \
    MessengerOverworld
import os
import importlib


def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_messenger_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    # Initialize a SQuAD teacher agent, which we will get context from
    module_name = 'parlai.tasks.squad.agents'
    class_name = 'DefaultTeacher'
    my_module = importlib.import_module(module_name)
    task_class = getattr(my_module, class_name)
    task_opt = opt.copy()
    task_opt['datatype'] = 'train'
    task_opt['datapath'] = opt['datapath']

    messenger_manager = MessengerManager(opt=opt)
    messenger_manager.setup_server()
    messenger_manager.init_new_state()

    def get_overworld(opt, agent):
        return MessengerOverworld(opt, agent)

    def assign_agent_role(agent):
        agent[0].disp_id = 'Agent'

    def run_conversation(manager, opt, agents, task_id):
        task = task_class(task_opt)
        agent = agents[0]
        world = QADataCollectionWorld(
            opt=opt,
            task=task,
            agent=agent
        )
        while not world.episode_done():
            world.parley()
        world.shutdown()

    # World with no onboarding
    messenger_manager.set_onboard_functions({'default': None})
    task_functions = {'default': run_conversation}
    assign_agent_roles = {'default': assign_agent_role}
    messenger_manager.set_agents_required({'default': 1})

    messenger_manager.set_overworld_func(get_overworld)
    messenger_manager.setup_socket()
    try:
        messenger_manager.start_new_run()
        messenger_manager.start_task(
            assign_role_functions=assign_agent_roles,
            task_functions=task_functions,
        )
    except BaseException:
        raise
    finally:
        messenger_manager.shutdown()


if __name__ == '__main__':
    main()

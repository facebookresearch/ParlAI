#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.messenger.tasks.overworld_demo.worlds import MessengerOverworld
from parlai.messenger.core.messenger_manager import MessengerManager

import os


def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_messenger_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt['password'] = 'ParlAI'  # If password is none anyone can chat

    messenger_manager = MessengerManager(opt=opt)
    messenger_manager.setup_server()
    messenger_manager.init_new_state()

    def get_overworld(opt, agent):
        return MessengerOverworld(opt, agent)

    onboard_functions = {name: worlds[0].run for (name, worlds)
                         in MessengerOverworld.DEMOS.items()}
    messenger_manager.set_onboard_functions(onboard_functions)
    task_functions = {name: worlds[1].run for (name, worlds)
                      in MessengerOverworld.DEMOS.items()}
    assign_agent_roles = {name: worlds[1].assign_roles for (name, worlds)
                          in MessengerOverworld.DEMOS.items()}
    agents_required = {name: worlds[1].MAX_AGENTS for (name, worlds)
                       in MessengerOverworld.DEMOS.items()}
    messenger_manager.set_agents_required(agents_required)

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

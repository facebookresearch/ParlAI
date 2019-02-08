#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.messenger.tasks.chatbot.worlds import \
    MessengerBotChatTaskWorld
from parlai.messenger.core.messenger_manager import MessengerManager
from parlai.messenger.core.worlds import SimpleMessengerOverworld as \
    MessengerOverworld
from parlai.core.agents import create_agent, create_agent_from_shared


def main():
    argparser = ParlaiParser(True, True)
    argparser.add_messenger_args()
    opt = argparser.parse_args()
    print(opt)
    if opt['model'] is None and opt['model_file'] is None:
        print("Model must be specified")
        return
    bot = create_agent(opt)
    shared_bot_params = bot.share()

    messenger_manager = MessengerManager(opt=opt)
    messenger_manager.setup_server()
    messenger_manager.init_new_state()

    def get_overworld(opt, agent):
        return MessengerOverworld(opt, agent)

    def assign_agent_role(agent):
        agent[0].disp_id = 'Agent'

    def run_conversation(manager, opt, agents, task_id):
        agent = agents[0]
        this_bot = create_agent_from_shared(shared_bot_params)

        world = MessengerBotChatTaskWorld(
            opt=opt,
            agent=agent,
            bot=this_bot
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

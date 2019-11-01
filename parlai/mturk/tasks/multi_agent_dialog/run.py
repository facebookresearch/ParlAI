#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.tasks.multi_agent_dialog.worlds import (
    MTurkMultiAgentDialogWorld,
    MTurkMultiAgentDialogOnboardWorld,
)
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.mturk.tasks.multi_agent_dialog.task_config import task_config


def main():
    """
    This task consists of one local human agent and two MTurk agents, each MTurk agent
    will go through the onboarding step to provide information about themselves, before
    being put into a conversation.

    You can end the conversation by sending a message ending with `[DONE]` from human_1.
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt.update(task_config)

    mturk_agent_1_id = 'mturk_agent_1'
    mturk_agent_2_id = 'mturk_agent_2'
    human_agent_1_id = 'human_1'
    mturk_agent_ids = [mturk_agent_1_id, mturk_agent_2_id]
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=mturk_agent_ids)
    mturk_manager.setup_server()

    try:
        mturk_manager.start_new_run()
        mturk_manager.create_hits()

        def run_onboard(worker):
            world = MTurkMultiAgentDialogOnboardWorld(opt=opt, mturk_agent=worker)
            while not world.episode_done():
                world.parley()
            world.shutdown()

        # You can set onboard_function to None to skip onboarding
        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        eligibility_function = {'func': check_worker_eligibility, 'multiple': False}

        def assign_worker_roles(workers):
            for index, worker in enumerate(workers):
                worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        def run_conversation(mturk_manager, opt, workers):
            # Create mturk agents
            mturk_agent_1 = workers[0]
            mturk_agent_2 = workers[1]

            # Create the local human agents
            human_agent_1 = LocalHumanAgent(opt={})
            human_agent_1.id = human_agent_1_id

            world = MTurkMultiAgentDialogWorld(
                opt=opt, agents=[human_agent_1, mturk_agent_1, mturk_agent_2]
            )

            while not world.episode_done():
                world.parley()

            world.shutdown()

        mturk_manager.start_task(
            eligibility_function=eligibility_function,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation,
        )

    except BaseException:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()

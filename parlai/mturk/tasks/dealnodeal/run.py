# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import os
import time
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.tasks.dealnodeal.worlds import \
    MTurkDealNoDealDialogWorld
from parlai.agents.local_human.local_human import LocalHumanAgent
from task_config import task_config
import copy
from itertools import product
from joblib import Parallel, delayed
import threading

"""
This task consists of one agent talking to an MTurk worker
"""
def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['task'] = 'dealnodeal'
    opt['datatype'] = 'valid'
    opt.update(task_config)

    mturk_agent_1_id = 'mturk_agent_1'
    mturk_agent_2_id = 'mturk_agent_2'
    human_agent_1_id = 'human_1'
    mturk_agent_ids = [mturk_agent_1_id] #, mturk_agent_2_id]
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids = mturk_agent_ids
    )

    mturk_manager.setup_server()

    try:
        mturk_manager.start_new_run()
        mturk_manager.create_hits()

        mturk_manager.set_onboard_function(onboard_function=None)
        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for index, worker in enumerate(workers):
                worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        def run_conversation(mturk_manager, opt, workers):
            # Create the local human agents
            human_agent_1 = LocalHumanAgent(opt=None)
            human_agent_1.id = human_agent_1_id
            opt["batchindex"] = mturk_manager.started_conversations

            world = MTurkDealNoDealDialogWorld(
                opt=opt,
                agents=[human_agent_1] + workers
            )

            while not world.episode_done():
                world.parley()

            world.shutdown()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )

    except:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()

if __name__ == '__main__':
    main()

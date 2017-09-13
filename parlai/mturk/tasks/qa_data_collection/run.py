# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.qa_data_collection.worlds import \
    QADataCollectionOnboardWorld, QADataCollectionWorld
from parlai.mturk.core.mturk_manager import MTurkManager
from task_config import task_config
import time
import os
import importlib
import copy
from itertools import product
from joblib import Parallel, delayed


def main():
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    opt = argparser.parse_args()
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt.update(task_config)

    # Initialize a SQuAD teacher agent, which we will get context from
    module_name = 'parlai.tasks.squad.agents'
    class_name = 'DefaultTeacher'
    my_module = importlib.import_module(module_name)
    task_class = getattr(my_module, class_name)
    task_opt = {}
    task_opt['datatype'] = 'train'
    task_opt['datapath'] = opt['datapath']

    mturk_agent_id = 'Worker'
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids = [mturk_agent_id]
    )
    mturk_manager.setup_server()

    def run_onboard(worker):
        world = QADataCollectionOnboardWorld(opt=opt, mturk_agent=worker)
        while not world.episode_done():
            world.parley()
        world.shutdown()

    mturk_manager.set_onboard_function(onboard_function=None)

    try:
        mturk_manager.start_new_run()
        mturk_manager.create_hits()

        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(worker):
            worker[0].id = mturk_agent_id

        global run_conversation
        def run_conversation(mturk_manager, opt, workers):
            task = task_class(task_opt)
            mturk_agent = workers[0]
            world = QADataCollectionWorld(
                opt=opt,
                task=task,
                mturk_agent=mturk_agent
            )
            while not world.episode_done():
                world.parley()
            world.shutdown()
            world.review_work()

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

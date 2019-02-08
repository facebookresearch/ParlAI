#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import RephrasePersonaWorld, PersonasGenerator
from task_config import task_config

import os


def main():
    """This task consists of one agent, model or MTurk worker, talking to an
    MTurk worker to negotiate a deal.
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument('-min_t', '--min_turns', default=5, type=int,
                           help='minimum number of turns')
    argparser.add_argument('-mt', '--max_turns', default=10, type=int,
                           help='maximal number of chat turns')
    argparser.add_argument('-mx_rsp_time', '--max_resp_time', default=150,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--ag_shutdown_time', default=120,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--persona-type', default='both', type=str,
                           choices=['both', 'self', 'other'],
                           help='Which personas to load from personachat')
    opt = argparser.parse_args()

    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)

    opt['extract_personas_path'] = os.path.join(opt['datapath'], opt['task'])
    opt.update(task_config)

    mturk_agent_ids = ['PERSON_1']

    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=mturk_agent_ids
    )

    mturk_manager.setup_server(task_directory_path=directory_path)

    personas_generator = PersonasGenerator(opt)
    opt['personas_generator'] = personas_generator

    try:
        mturk_manager.start_new_run()
        mturk_manager.create_hits()

        if not opt['is_sandbox']:
            # ADD BLOCKED WORKERS HERE
            blocked_worker_list = []
            for w in blocked_worker_list:
                mturk_manager.block_worker(
                    w,
                    'We found that you have unexpected behaviors in our '
                    'previous HITs. For more questions please email us.'
                )

        def run_onboard(worker):
            pass

        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for index, worker in enumerate(workers):
                worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        def run_conversation(mturk_manager, opt, workers):
            worker = workers[0]
            world = RephrasePersonaWorld(opt, worker)
            while not world.episode_done():
                world.parley()
            world.save_data()
            world.shutdown()
            world.review_work()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )

    except BaseException:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()

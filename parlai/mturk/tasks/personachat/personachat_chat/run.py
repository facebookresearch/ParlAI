#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import \
    PersonaChatWorld, PersonaProfileWorld, PersonasGenerator
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
    argparser.add_argument('-mx_psn_time', '--max_persona_time', type=int,
                           default=300, help='time limit for turker'
                           'entering the persona')
    argparser.add_argument('--ag_shutdown_time', default=120,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--persona-type', default='both', type=str,
                           choices=['both', 'self', 'other'],
                           help='Which personas to load from personachat')
    argparser.add_argument('--revised', default=False, type='bool',
                           help='Whether to use revised personas')
    argparser.add_argument('-rt', '--range_turn', default='5,7',
                           help='sample range of number of turns')
    argparser.add_argument('--personas-path', default=None,
                           help='specify path for personas data')
    opt = argparser.parse_args()

    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)

    if not opt.get('personas_path'):
        opt['personas_path'] = (
            argparser.parlai_home + '/parlai/mturk/personachat_chat/data'
        )

    opt.update(task_config)

    opt['extract_personas_path'] = os.path.join(opt['datapath'], 'personachat_chat')

    mturk_agent_ids = ['PERSON_1', 'PERSON_2']

    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=mturk_agent_ids
    )

    persona_generator = PersonasGenerator(opt)
    mturk_manager.setup_server(task_directory_path=directory_path)

    try:
        mturk_manager.start_new_run()
        mturk_manager.create_hits()

        if not opt['is_sandbox']:
            blocked_worker_list = []
            for w in blocked_worker_list:
                mturk_manager.block_worker(
                    w,
                    'We found that you have unexpected behaviors in our previous '
                    'HITs. For more questions please email us.'
                )

        def run_onboard(worker):
            worker.persona_generator = persona_generator
            world = PersonaProfileWorld(opt, worker)
            world.parley()
            world.shutdown()
        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for index, worker in enumerate(workers):
                worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        def run_conversation(mturk_manager, opt, workers):
            agents = [workers[0], workers[1]]
            conv_idx = mturk_manager.conversation_index
            world = PersonaChatWorld(
                opt=opt,
                agents=agents,
                range_turn=[int(s) for s in opt['range_turn'].split(',')],
                max_turn=opt['max_turns'],
                max_resp_time=opt['max_resp_time'],
                world_tag='conversation t_{}'.format(conv_idx)
            )
            world.reset_random()
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

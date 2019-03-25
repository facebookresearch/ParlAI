#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import \
    Convai2EvalWorld, PersonaProfileWorld, PersonasGenerator
from task_config import task_config
import time

import os

MASTER_QUALIF = {
    'QualificationTypeId': '2F1QJWKUDD8XADTFD2Q0G6UTO95ALH',
    'Comparator': 'Exists',
    'RequiredToPreview': True
}

MASTER_QUALIF_SDBOX = {
    'QualificationTypeId': '2ARFPLSP75KLA8M8DH1HTEQVJT3SY6',
    'Comparator': 'Exists',
    'RequiredToPreview': True
}


def main():
    """This task consists of an MTurk agent evaluating a chit-chat model. They
    are asked to chat to the model adopting a specific persona. After their
    conversation, they are asked to evaluate their partner on several metrics.
    """
    argparser = ParlaiParser(False, add_model_args=True)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument('-mt', '--max-turns', default=10, type=int,
                           help='maximal number of chat turns')
    argparser.add_argument('--max-resp-time', default=240,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--max-persona-time', type=int,
                           default=300, help='time limit for turker'
                           'entering the persona')
    argparser.add_argument('--ag-shutdown-time', default=120,
                           type=int,
                           help='time limit for entering a dialog message')
    argparser.add_argument('--persona-type', default='both', type=str,
                           choices=['both', 'self', 'other'],
                           help='Which personas to load from personachat')
    argparser.add_argument('--revised', default=False, type='bool',
                           help='Whether to use revised personas')
    argparser.add_argument('-rt', '--range-turn', default='5,6',
                           help='sample range of number of turns')
    argparser.add_argument('--auto-approve-delay', type=int,
                           default=3600 * 24 * 1,
                           help='how long to wait for auto approval')
    argparser.add_argument('--only-masters', type='bool', default=False,
                           help='Set to True to use only master turks for this' +
                                ' test eval, default is %(default)s')

    # ADD MODEL ARGS HERE, UNCOMMENT TO USE KVMEMNN MODEL AS AN EXAMPLE
    # argparser.set_defaults(
    #     model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
    #     model_file='models:convai2/kvmemnn/model',
    # )

    opt = argparser.parse_args()

    # add additional model args
    opt['override'] = {
        'no_cuda': True,
        'interactive_mode': True,
        'tensorboard_log': False
    }

    bot = create_agent(opt)
    shared_bot_params = bot.share()
    print(
        '=== Actual bot opt === :\n {}'.format(
            '\n'.join(["[{}] : {}".format(k, v) for k, v in bot.opt.items()])
        )
    )
    folder_name = (
        'master_{}_YOURCOMMENT__'.format(opt['only_masters']) +
        '__'.join(['{}_{}'.format(k, v) for k, v in opt['override'].items()])
    )

    #  this is mturk task, not convai2 task from ParlAI
    opt['task'] = 'convai2:self'
    if 'data_path' not in opt:
        opt['data_path'] = os.getcwd() + '/data/' + folder_name
    opt.update(task_config)

    mturk_agent_ids = ['PERSON_1']

    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=mturk_agent_ids
    )

    persona_generator = PersonasGenerator(opt)
    mturk_manager.setup_server()

    try:
        mturk_manager.start_new_run()
        agent_qualifications = []
        if opt['only_masters']:
            if opt['is_sandbox']:
                agent_qualifications.append(MASTER_QUALIF_SDBOX)
            else:
                agent_qualifications.append(MASTER_QUALIF)
        mturk_manager.ready_to_accept_workers()
        mturk_manager.create_hits(qualifications=agent_qualifications)

        if not opt['is_sandbox']:
            # ADD SOFT-BLOCKED WORKERS HERE
            # NOTE: blocking qual *must be* specified
            blocked_worker_list = []
            for w in blocked_worker_list:
                print('Soft Blocking {}\n'.format(w))
                mturk_manager.soft_block_worker(w)
                time.sleep(0.1)  # do the sleep to prevent amazon query drop

        def run_onboard(worker):
            worker.persona_generator = persona_generator
            world = PersonaProfileWorld(opt, worker)
            world.parley()
            world.shutdown()
        mturk_manager.set_onboard_function(onboard_function=run_onboard)

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            for index, worker in enumerate(workers):
                worker.id = mturk_agent_ids[index % len(mturk_agent_ids)]

        def run_conversation(mturk_manager, opt, workers):
            agents = workers[0]
            conv_idx = mturk_manager.conversation_index
            world = Convai2EvalWorld(
                opt=opt,
                agents=[agents],
                range_turn=[int(s) for s in opt['range_turn'].split(',')],
                max_turn=opt['max_turns'],
                max_resp_time=opt['max_resp_time'],
                model_agent_opt=shared_bot_params,
                world_tag='conversation t_{}'.format(conv_idx),
                agent_timeout_shutdown=opt['ag_shutdown_time'],
            )
            world.reset_random()
            while not world.episode_done():
                world.parley()
            world.save_data()

            world.shutdown()

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

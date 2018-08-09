# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.mturk.core.mturk_manager import MTurkManager
from worlds import \
    Convai2EvalWorld, PersonaProfileWorld, PersonasGenerator
from task_config import task_config

import os


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
    argparser.add_argument('--max-resp-time', default=180,
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
                           default=3600*24*1, help='how long to wait for  \
                           auto approval')

    # ADD MODEL ARGS HERE (KVMEMNN ADDED AS AN EXAMPLE)
    argparser.set_defaults(
        model='projects.personachat.kvmemnn.kvmemnn:Kvmemnn',
        model_file='models:convai2/kvmemnn/model',
    )
    opt = argparser.parse_args()

    # add additional model args
    opt['no_cuda'] = True
    opt['override'] = {'interactive_mode': True}
    opt['interactive_mode'] = True
    opt['model_name'] = 'kvmemnn_convai2'

    bot = create_agent(opt)
    shared_bot_params = bot.share()

    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    if 'data_path' not in opt:
        opt['data_path'] = os.getcwd() + '/data/' + opt['task']
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
        mturk_manager.create_hits()

        if not opt['is_sandbox']:
            # ADD BLOCKED WORKERS HERE
            blocked_worker_list = []
            for w in blocked_worker_list:
                mturk_manager.block_worker(
                    w,
                    'We found that you have unexpected behaviors in our \
                     previous HITs. For more questions please email us.')

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

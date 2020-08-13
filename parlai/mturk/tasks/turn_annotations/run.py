#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import time
import threading
from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.turn_annotations.constants import (
    AGENT_0,
    ANNOTATIONS_CONFIG,
    TASK_CONFIG,
    LEFT_PANE_TEXT,
    FINAL_RATING_QUESTION,
)
from parlai.mturk.tasks.turn_annotations.worlds import (
    TurnAnnotationsOnboardWorld,
    TurnAnnotationsChatWorld,
)
from parlai.mturk.tasks.turn_annotations.bot_agent import TurkLikeAgent


def run_task(override_opt):
    """
    This task consists of an MTurk worker talking to a model and MTurker also evaluates
    each utterance of the bot for various buckets (see constants).
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '-num_t', '--num_turns', default=6, type=int, help='minimum number of turns'
    )
    argparser.add_argument(
        '--task-model-parallel',
        default=True,
        type=bool,
        help='Whether to load models to be used with model_parallel True.',
    )
    argparser.add_argument(
        '--auto-approve-delay',
        dest='auto_approve_delay',
        type=int,
        default=3600 * 24 * 5,
        help='how long to wait for auto approval',
    )
    argparser.add_argument(
        '--max-resp-time',
        type=int,
        default=180,
        help='time limit for entering a dialog message',
    )
    argparser.add_argument(
        '--max-onboard-time',
        type=int,
        default=300,
        help='time limit accepting onboarding',
    )
    argparser.add_argument(
        '--base-save-folder',
        default=None,
        type=str,
        help='base folder for saving all crowdsourcing results',
    )
    argparser.add_argument(
        '--base-model-folder',
        default=None,
        type=str,
        help='base folder for loading model files from',
    )
    argparser.add_argument(
        '--onboard-worker-answer-folder',
        default=None,
        type=str,
        help='base folder for saving all worker answer results during onboarding',
    )

    argparser.set_params(**override_opt)
    opt = argparser.parse_args()

    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)

    opt['left_pane_text'] = LEFT_PANE_TEXT
    opt['final_rating_question'] = FINAL_RATING_QUESTION
    opt.update(TASK_CONFIG)

    # NOTE: you have to set all three of these opts to enforce the MTurk core
    # param max_hits_per_worker.
    #  - Without unique_qual_name, MTurkManager creates different qualification
    #    for each run (so a worker could do N hits per run) Also, the
    #    worker has to get to N HITs in at least one run or they won't be given
    #    the qualification.
    #  - allowed_conversations is like max concurrent conversations
    #    allowed_conversations needs to be 1 or the actual max would be N +
    #    allowed_conversations. Worker gets notified via frontend message that
    #    they aren't eligible (second description screen), UNLESS the frontend
    #    overwrites that functionality.
    # There's also still a race condition where the worker might be able to open
    # 1 extra task
    opt['unique_qual_name'] = 'turn_annotations_max_submissions'
    opt['max_hits_per_worker'] = 10
    opt['allowed_conversations'] = 3

    # Limits the number of models that can generate at once
    MAX_CONCURRENT_RESPONSES = 1
    semaphore = threading.Semaphore(MAX_CONCURRENT_RESPONSES)

    run_statistics = copy.deepcopy(opt['conversations_needed'])
    run_statistics = {r: 0 for (r, v) in run_statistics.items()}
    onboard_statistics = {}

    save_folder = 'sandbox' if opt['is_sandbox'] else 'live'
    opt['save_folder'] = os.path.join(
        opt['base_save_folder'], save_folder, time.strftime("%Y_%m_%d")
    )
    os.makedirs(opt['save_folder'], exist_ok=True)

    print(
        f'Going to start collecting {opt["num_conversations"]} conversations, max_hits_per_worker: {opt["max_hits_per_worker"]}, reward: {opt["reward"]}, is_sandbox: {opt["is_sandbox"]}.'
    )

    # Create the models before it launches Heroku backend b/c takes a while
    models_needed = list(opt['conversations_needed'].keys())
    active_models = [m for m in models_needed if opt['conversations_needed'][m] > 0]
    shared_bot_agents = TurkLikeAgent.get_bot_agents(
        opt, active_models, datapath=opt['datapath']
    )

    mturk_agent_ids = [AGENT_0]
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=mturk_agent_ids)
    mturk_manager.setup_server(task_directory_path=directory_path)

    try:
        mturk_manager.start_new_run()
        mturk_manager.create_hits()

        def run_onboard(worker):
            world = TurnAnnotationsOnboardWorld(opt, worker)
            status = world.parley()
            if status not in onboard_statistics:
                onboard_statistics[status] = 0
            onboard_statistics[status] += 1
            print(
                f'After onboard world parley. About to shutdown onboard world for {worker.worker_id}, status was: {status}. Total onboard statistics for this run are: {onboard_statistics}.'
            )
            world.shutdown()

        mturk_manager.set_onboard_function(onboard_function=run_onboard)
        mturk_manager.ready_to_accept_workers()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            workers[0].id = mturk_agent_ids[0]

        def run_conversation(mturk_manager, opt, workers):
            remaining_counts_needed = [
                (m, c - run_statistics[m])
                for (m, c) in opt['conversations_needed'].items()
            ]
            remaining_counts_needed.sort(reverse=True, key=lambda x: x[1])
            model_name = remaining_counts_needed[0][0]
            print(f'Remaining conversation counts needed: {remaining_counts_needed}')

            # Get a bot and add it to the list of "workers"
            print(f'Choosing the "{model_name}" model for the bot.')
            agent = create_agent_from_shared(shared_bot_agents[model_name])
            bot_worker = TurkLikeAgent(
                opt,
                model_name=model_name,
                model_agent=agent,
                num_turns=opt['num_turns'],
                semaphore=semaphore,
            )
            workers_including_bot = workers + [bot_worker]

            assert len(workers_including_bot) == 2

            conv_idx = mturk_manager.conversation_index
            world = TurnAnnotationsChatWorld(
                opt=opt,
                agents=workers_including_bot,
                num_turns=opt['num_turns'],
                max_resp_time=opt['max_resp_time'],
                tag='conversation t_{}'.format(conv_idx),
                annotations_config=ANNOTATIONS_CONFIG,
            )
            while not world.episode_done():
                print('About to parley')
                world.parley()
            model_nickname, convo_finished = world.save_data()

            if convo_finished:
                run_statistics[model_nickname] += 1

            world.shutdown()
            world.review_work()

        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation,
        )

    except BaseException:
        raise
    finally:
        mturk_manager.expire_all_unassigned_hits()
        mturk_manager.shutdown()

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
import threading
import time
from typing import Optional

from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.turn_annotations.constants import AGENT_0
from parlai.mturk.tasks.turn_annotations.worlds import (
    TurnAnnotationsChatWorld,
    TurnAnnotationsOnboardWorld,
)
from parlai.mturk.tasks.turn_annotations.bot_agent import TurkLikeAgent
from parlai.tasks.blended_skill_talk.agents import ContextGenerator


def run_task(override_opt: Optional[dict] = None):
    """
    This task consists of an MTurk worker talking to a model and MTurker also evaluates
    each utterance of the bot for various buckets (see constants).
    """

    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'task_config'
    )
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    default_task_folder = os.path.join(
        argparser.parlai_home, 'data', 'turn_annotations'
    )
    argparser.add_mturk_args()
    argparser.add_argument(
        '-num_t', '--num_turns', default=6, type=int, help='minimum number of turns'
    )
    argparser.add_argument(
        '--conversations-needed',
        dest='conversations_needed_string',
        default=None,
        type=str,
        help='Number of convos needed for each model. For example: "modelA:50,modelB:20"',
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
        default=default_task_folder,
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
        default=os.path.join(default_task_folder, 'onboard_answers'),
        type=str,
        help='base folder for saving all worker answer results during onboarding',
    )
    argparser.add_argument(
        '--worker-blocklist-paths',
        default=None,
        type=str,
        help='Path(s) to a list of IDs of workers to soft-block, separated by newlines. Use commas to indicate multiple lists',
    )
    argparser.add_argument(
        '--check-acceptability',
        default=False,
        type=bool,
        help="Check worker's responses against several metrics of acceptability",
    )
    argparser.add_argument(
        '--include-persona', default=False, type=bool, help="Show persona to the bot"
    )
    argparser.add_argument(
        '--conversation-start-mode',
        default='hi',
        type=str,
        choices=['hi', 'bst'],
        help='Whether to show "Hi!" or two previous utterances (as in BlendedSkillTalk) at the beginning of the conversation',
    )
    argparser.add_argument(
        '--context-seed',
        default=None,
        type=int,
        help="Set seed for pulling the context info (for testing)",
    )
    argparser.add_argument(
        '--hit-config-path',
        default=os.path.join(config_folder, 'hit_config.json'),
        type=str,
        help='Path to file of parameters describing how MTurk will describe the HIT to the workers',
    )
    argparser.add_argument(
        '--task-description-path',
        default=os.path.join(config_folder, 'task_description.html'),
        type=str,
        help='Path to file of HTML to show on the task-description page',
    )
    argparser.add_argument(
        '--left-pane-text-path',
        default=os.path.join(config_folder, 'left_pane_text.html'),
        type=str,
        help='Path to file of HTML to show on the left-hand pane of the chat window',
    )
    argparser.add_argument(
        '--annotations-intro',
        default='Does this comment from your partner have any of the following attributes? (Check all that apply)',
        type=str,
        help='Text shown to worker before they fill out annotation form',
    )
    argparser.add_argument(
        '--annotations-config-path',
        default=os.path.join(config_folder, 'annotations_config.json'),
        type=str,
        help='Path to JSON of annotation categories',
    )
    argparser.add_argument(
        '--onboard-task-data-path',
        default=os.path.join(config_folder, 'onboard_task_data.json'),
        type=str,
        help='Path to JSON containing settings for running onboarding',
    )
    argparser.add_argument(
        '--final-rating-question',
        default='Please rate your partner on a scale of 1-5.',
        type=str,
        help='Text to show when asking worker to make their final rating',
    )

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
    argparser.set_defaults(
        unique_qual_name='turn_annotations_max_submissions',
        max_hits_per_worker=10,
        allowed_conversations=3,
    )

    if override_opt is not None:
        argparser.set_params(**override_opt)
        opt = argparser.parse_args([])
    else:
        opt = argparser.parse_args()
    directory_path = os.path.dirname(os.path.abspath(__file__))
    opt['task'] = os.path.basename(directory_path)

    # Set the number of conversations needed
    if opt.get('conversations_needed_string') is not None:
        parts = opt['conversations_needed_string'].split(',')
        conversations_needed = {}
        for part in parts:
            model_name, num_string = part.split(':')
            conversations_needed[model_name] = int(num_string)
        opt['conversations_needed'] = conversations_needed

    # Read in workers to soft-block
    if opt.get('worker_blocklist_paths') is not None:
        blocklist_paths = opt['worker_blocklist_paths'].split(',')
        worker_blocklist = set()
        for path in blocklist_paths:
            with open(path) as f:
                worker_blocklist |= set(f.read().strip().split('\n'))
        opt['worker_blocklist'] = worker_blocklist

    # Read in and define text shown to users
    if opt.get('hit_config') is None:
        with open(opt['hit_config_path']) as f:
            opt['hit_config'] = json.load(f)
        opt.update(opt['hit_config'])
        # Add all of the settings in hit_config into the base opt
    if opt.get('task_description') is None:
        with open(opt['task_description_path']) as f:
            opt['task_description'] = f.readlines()
    if opt.get('left_pane_text') is None:
        with open(opt['left_pane_text_path']) as f:
            opt['left_pane_text'] = f.readlines()
    if opt.get('annotations_config') is None:
        with open(opt['annotations_config_path']) as f:
            opt['annotations_config'] = json.load(f)
    if opt.get('onboard_task_data') is None:
        with open(opt['onboard_task_data_path']) as f:
            opt['onboard_task_data'] = json.load(f)

    # Limits the number of models that can generate at once
    max_concurrent_responses = 1
    semaphore = threading.Semaphore(max_concurrent_responses)

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
    shared_bot_agents = TurkLikeAgent.get_bot_agents(opt, active_models)

    mturk_agent_ids = [AGENT_0]
    mturk_manager = MTurkManager(opt=opt, mturk_agent_ids=mturk_agent_ids)
    mturk_manager.setup_server(task_directory_path=directory_path)

    if opt['include_persona'] or opt['conversation_start_mode'] == 'bst':
        context_generator = ContextGenerator(opt, datatype='test', seed=0)
        # We pull from the test set so that the model can't regurgitate
        # memorized conversations
    else:
        context_generator = None

    try:
        mturk_manager.start_new_run()
        mturk_manager.create_hits()

        if not opt['is_sandbox']:
            # Soft-block all chosen workers
            if len(opt['worker_blocklist']) > 0:
                print(f"About to soft-block {len(opt['worker_blocklist'])} workers.")
                for w in set(opt['worker_blocklist']):
                    try:
                        print('Soft Blocking {}\n'.format(w))
                        mturk_manager.soft_block_worker(w)
                    except Exception as e:
                        print(f'Did not soft block worker {w}: {e}')
                    time.sleep(0.1)
            else:
                print(
                    'WARNING: We are in live mode, but a list of workers to soft-block '
                    'has not been passed in.'
                )

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

            # Get context: personas, previous utterances, etc.
            if context_generator is not None:
                context_info = context_generator.get_context()
            else:
                context_info = None

            conv_idx = mturk_manager.conversation_index
            world = TurnAnnotationsChatWorld(
                opt=opt,
                agents=workers_including_bot,
                num_turns=opt['num_turns'],
                max_resp_time=opt['max_resp_time'],
                tag='conversation t_{}'.format(conv_idx),
                context_info=context_info,
            )
            while not world.episode_done():
                print('About to parley')
                world.parley()
            model_nickname, worker_is_unacceptable, convo_finished = world.save_data()
            if worker_is_unacceptable:
                print(f'Soft-blocking worker {workers[0].worker_id}')
                mturk_manager.soft_block_worker(workers[0].worker_id)
                time.sleep(0.1)
            if not worker_is_unacceptable and convo_finished:
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


if __name__ == '__main__':
    run_task()

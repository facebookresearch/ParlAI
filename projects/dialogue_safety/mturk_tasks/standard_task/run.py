#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import MTurkManager
import parlai.mturk.core.mturk_utils as mutils

from task_config import task_config
from projects.dialogue_safety.mturk_tasks.standard_task.worlds import (
    SingleTurnSafetyGenScratch,
    SingleTurnSafetyGenTopic,
    SingleTurnSafetyGenOnboardingWorld,
    BLOCK_QUALIFICATION,
)

import os
import random
import threading
import pickle
import json


ParlaiParser()  # instantiate to set PARLAI_HOME environment var
DEFAULT_SAVE_DIR = os.path.join(os.environ['PARLAI_HOME'], 'data')


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class GetTopics:
    def __init__(self, opt):
        self.file_name = opt.get('topics_file')
        self.topics = []
        self.total_turns = opt.get('num_per_eval')

        with open(self.file_name, 'r') as f:
            data = json.loads(f.read())
        self.topics = data['train']

    def random_topics(self, total_turns=None):
        # Returns list of topics of length total_turns
        num = total_turns if total_turns else self.total_turns
        return random.sample(self.topics, num)


class TrackOnboardingCompletion(object):
    def __init__(self, opt):
        save_dir = opt.get('save_dir')
        version_num = opt.get('version_num', 0)
        self.list_path = os.path.join(
            save_dir, 'completed_onboarding_v{}.txt'.format(version_num)
        )
        self.blocked_path = os.path.join(
            save_dir, 'blocked_onboarding_v{}.txt'.format(version_num)
        )
        ensure_dir(self.list_path)
        self.completed = self.load_completion_list()
        self.soft_blocked = self.load_soft_blocked_list()
        self.list_lock = threading.RLock()
        self.block_lock = threading.RLock()

    def did_worker_complete(self, worker_id):
        with self.list_lock:
            return True if worker_id in self.completed else False

    def mark_worker_completed(self, worker_id):
        with self.list_lock:
            if worker_id not in self.completed:
                self.completed.append(worker_id)
            # now save list
        self.save_completion_list()

    def save_completion_list(self):
        print('Saving onboarding completion list to {}.'.format(self.list_path))
        with self.list_lock:
            with open(self.list_path, 'wb') as g:
                pickle.dump(self.completed, g)

    def remove_worker_from_completion_list(self, worker_id):
        with self.list_lock:
            if worker_id in self.completed:
                self.completed.remove(worker_id)

    def load_completion_list(self):
        if os.path.isfile(self.list_path):
            with open(self.list_path, 'rb') as f:
                completed = pickle.load(f)
            return completed
        return []

    def mark_worker_blocked(self, worker_id):
        with self.block_lock:
            self.soft_blocked.append(worker_id)
        # save list
        self.save_blocked_list()

    def save_blocked_list(self):
        print('Saving blocked list to {}.'.format(self.blocked_path))
        with self.block_lock:
            with open(self.blocked_path, 'wb') as g:
                pickle.dump(self.soft_blocked, g)

    def load_soft_blocked_list(self):
        if os.path.isfile(self.blocked_path):
            with open(self.blocked_path, 'rb') as f:
                blocked = pickle.load(f)
            return blocked
        return []


def main():
    argparser = ParlaiParser(False, add_model_args=True)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '--run-onboard', type='bool', default=True, help='run onboard world'
    )
    argparser.add_argument(
        '--save-dir',
        type=str,
        default=os.path.join(DEFAULT_SAVE_DIR, 'mturk_safety_gen_data'),
        help='where to save onboard tracking data',
    )
    argparser.add_argument(
        '--topics-file',
        type=str,
        default=os.path.join(
            DEFAULT_SAVE_DIR, 'wizard_of_wikipedia', 'topic_splits.json'
        ),
        help='topics data',
    )
    argparser.add_argument(
        '--num-per-eval', type=int, default=5, help='number of sentences per HIT'
    )
    argparser.add_argument(
        '--ok-or-notok',
        type=str,
        default='NOT OK',
        choices=['OK', 'NOT OK'],
        help='ask turker to generate messages that are' 'either OK or NOT OK',
    )
    argparser.add_argument(
        '--len-range',
        type=str,
        default='4,20',
        help='range to enforce minimum and maximum' 'submitted sentence lengths',
    )
    argparser.add_argument(
        '--auto-approve-delay',
        type=int,
        default=3600 * 24 * 2,
        help='how long to wait for auto approval (default ' 'is two days)',
    )
    opt = argparser.parse_args()

    # Set the task name to be the folder name
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    # append the contents of task_config.py to the configuration
    opt.update(task_config)

    # load topics
    if not os.path.isfile(opt['topics_file']):
        # check for topics file
        from parlai.tasks.wizard_of_wikipedia.build import build

        print('[ Downloading topics data... ]')
        build(opt)
    print('[ Building Topics manager... ]')
    topics = GetTopics(opt)

    # Select an agent_id that worker agents will be assigned in their world
    mturk_agent_roles = ['Evaluator']

    mturk_manager = MTurkManager(
        opt=opt, mturk_agent_ids=mturk_agent_roles, use_db=True,
    )

    mturk_manager.setup_server(
        task_directory_path=os.path.dirname(os.path.abspath(__file__))
    )

    onboarding_tracker = TrackOnboardingCompletion(opt)

    def run_onboard(worker):
        nonlocal onboarding_tracker
        if onboarding_tracker.did_worker_complete(worker.worker_id):
            return
        else:
            role = mturk_agent_roles[0]
            worker.update_agent_id('Onboarding {}'.format(role))
            world = SingleTurnSafetyGenOnboardingWorld(
                opt=opt, mturk_agent=worker, onboarding_tracker=onboarding_tracker,
            )
            while not world.episode_done():
                world.parley()
            world.shutdown()
            onboarding_tracker.mark_worker_completed(worker.worker_id)
            return world.prep_save_data([worker])

    if opt.get('run_onboard'):
        mturk_manager.set_onboard_function(onboard_function=run_onboard)

    try:
        # Initialize run information
        mturk_manager.start_new_run()

        # Set up the sockets and threads to recieve workers
        mturk_manager.ready_to_accept_workers()

        # Create a qualification to ensure a worker won't repeat modifying
        # sentences will become necessary toward the end of the stack

        qual_name = BLOCK_QUALIFICATION
        qual_desc = (
            'Qualification to ensure worker does not exceed maximum turns '
            'on this HIT'
        )
        qualification_id = mutils.find_or_create_qualification(
            qual_name, qual_desc, False, must_be_owned=False
        )
        max_qualif = {
            'QualificationTypeId': qualification_id,
            'Comparator': 'DoesNotExist',
            'RequiredToPreview': True,
        }
        qualifications = [max_qualif]

        mturk_manager.create_hits(qualifications=qualifications)

        def check_workers_eligibility(workers):
            return workers

        def assign_worker_roles(workers):
            for worker in workers:
                worker.id = mturk_agent_roles[0]

        def run_conversation(mturk_manager, opt, workers):
            worker = workers[0]
            worker.task_world_assignment = random.randint(1, 2)

            if worker.task_world_assignment == 1:
                worker.update_agent_id('Scratch')
                world = SingleTurnSafetyGenScratch(opt=opt, mturk_agents=workers,)
            else:
                worker.update_agent_id('Topic')
                world = SingleTurnSafetyGenTopic(
                    opt=opt, mturk_agents=workers, topics=topics.random_topics(),
                )

            while not world.episode_done():
                world.parley()

            world.shutdown()

            # Return the contents for saving
            return world.prep_save_data(workers)

        mturk_manager.start_task(
            eligibility_function=check_workers_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation,
        )

    except Exception:
        raise

    finally:
        onboarding_tracker.save_completion_list()
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()

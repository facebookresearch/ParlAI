#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.params import ParlaiParser
from projects.dialogue_safety.mturk_tasks.adversarial_task.worlds import (
    AdversarialSafetyGenScratch,
    AdversarialSafetyGenTopic,
    AdversarialOnboardingWorld,
    CLASS_OK,
    CLASS_NOT_OK,
)
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.agents import create_agent_from_model_file
from projects.dialogue_safety.mturk_tasks.adversarial_task.task_config import task_config
import parlai.mturk.core.mturk_utils as mutils
import projects.dialogue_safety.mturk_tasks.standard_task.run as sts_run
import projects.dialogue_safety.mturk_tasks.adversarial_task.model_configs as mc

import os
import random
import json

ParlaiParser()  # instantiate to set PARLAI_HOME environment var
DEFAULT_SAVE_DIR = os.path.join(os.environ['PARLAI_HOME'], 'data')


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


def main():
    argparser = ParlaiParser(False, add_model_args=True)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '--run-onboard', type='bool', default=True, help='run onboard to as a test'
    )
    argparser.add_argument(
        '--save-dir',
        type=str,
        default=DEFAULT_SAVE_DIR,
        help='where to save partial data',
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
        '--num-tries',
        type=int,
        default=2,
        help='number of tries to beat classifer per sentence',
    )
    argparser.add_argument(
        '--false-positive',
        type='bool',
        default=False,
        help='Beat classifier by either generating false \
                           positives or false negatives',
    )
    argparser.add_argument(
        '--auto-approve-delay',
        type=int,
        default=3600 * 24 * 4,
        help='how long to wait for auto approval (default ' 'is two days)',
    )
    opt = {}
    shared_param_list = []
    save_names = []
    for m in mc.models:
        argparser.set_params(
            model=m.get('model'), model_file=m.get('model_file'),
        )

        opt = argparser.parse_args()

        # add additional model args
        opt['override'] = {
            'no_cuda': True,
            'interactive_mode': True,
            'threshold': m.get('threshold'),
            'classes': [CLASS_NOT_OK, CLASS_OK],
            'classes_from_file': None,
        }
        for k, v in m.items():
            opt['override'][k] = v
        for k, v in opt['override'].items():
            opt[k] = v

        # Set the task name to be the folder name
        opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

        # append the contents of task_config.py to the configuration
        opt.update(task_config)

        bot = create_agent_from_model_file(opt['model_file'], opt['override'])
        shared_bot_params = bot.share()
        print(
            '=== Actual bot opt === :\n {}'.format(
                '\n'.join(["[{}] : {}".format(k, v) for k, v in bot.opt.items()])
            )
        )
        shared_param_list.append(shared_bot_params)
        save_names.append(m.get('save_name'))

    # set up topics
    topics = GetTopics(opt)

    # Select an agent_id that worker agents will be assigned in their world
    mturk_agent_roles = ['Evaluator']

    mturk_manager = MTurkManager(
        opt=opt, mturk_agent_ids=mturk_agent_roles, use_db=True,
    )

    mturk_manager.setup_server(
        task_directory_path=os.path.dirname(os.path.abspath(__file__))
    )

    onboarding_tracker = sts_run.TrackOnboardingCompletion(opt)

    # Create an onboard_function, which will be run for workers who have
    # accepted your task and must be completed before they are put in the
    # queue for a task world.
    def run_onboard(worker):
        nonlocal onboarding_tracker
        if onboarding_tracker.did_worker_complete(worker.worker_id):
            return
        else:
            role = mturk_agent_roles[0]
            worker.update_agent_id('Onboarding {}'.format(role))
            world = AdversarialOnboardingWorld(
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

        qual_name = ''  # come up with your own block qualification
        qual_desc = 'Qualification to ensure worker does not repeat evaluations.'
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
                world = AdversarialSafetyGenScratch(
                    opt=opt,
                    mturk_agents=workers,
                    model_agents_opts=shared_param_list,
                    save_names=save_names,
                )
            else:
                worker.update_agent_id('Topic')
                world = AdversarialSafetyGenTopic(
                    opt=opt,
                    topics=topics.random_topics(),
                    mturk_agents=workers,
                    model_agents_opts=shared_param_list,
                    save_names=save_names,
                )

            while not world.episode_done():
                world.parley()

            world.shutdown()
            world.review_work()

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

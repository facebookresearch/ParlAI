#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.params import ParlaiParser
from parlai.mturk.tasks.light.light_chat_eval.worlds import (
    LightEvalTestWorld, LightEvalTaskWorld
)
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.mturk.tasks.light.light_chat_eval.task_config import task_config
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
import os
import random


def main():
    '''Main script for running an eval task against the LIGHT dataset.

    special CLI arguments are
      --light-eval-task-type [speech, emote, action]
      --light-eval-unseen [False, True]

    This launches a task that, on a workers first attempt pairs with an entry
    from the training set. Then based on if the worker performs above a
    specified benchmark, they will either be soft blocked from evaluating or
    allowed to try against the test set.
    '''
    # Get relevant arguments
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.set_defaults(datatype='test:stream')
    argparser.add_argument(
        '--light-eval-task-type', default='speech',
        help='Type of task to be evaluating')
    argparser.add_argument(
        '--light-eval-unseen', default=False, type='bool',
        help='Evaluate against the unseen test rather than the seen test')
    opt = argparser.parse_args()

    task_opt = opt.copy()
    task_opt['task'] = 'light_dialog'
    assert opt['light_eval_task_type'] in ['speech', 'emote', 'action'], (
        '--light-eval-task-type must be one of speech, emote, or action'
    )
    LABEL_TYPE = opt['light_eval_task_type']  # speech, emote, action
    TRAIN_TURNS = 7
    TRAININGS = 1
    MAX_WRONG = 1
    if LABEL_TYPE != 'speech':
        TRAIN_TURNS = 3
        TRAININGS = 2
        MAX_WRONG = 3 if LABEL_TYPE == 'emote' else 2
    task_opt['light_label_type'] = LABEL_TYPE
    task_opt['light_use_action'] = 'all'
    task_opt['light_use_cands'] = '20'
    task_opt['light_use_emote'] = 'all'
    task_opt['light_use_objects'] = True
    task_opt['light_use_person_names'] = True
    task_opt['light_use_persona'] = 'self'
    task_opt['light_use_repeat'] = 'none'
    task_opt['light_use_setting'] = True
    task_opt['light_use_speech'] = 'all'
    task_opt['light_use_current_self_output'] = 'all'
    task_opt['light_use_clip_cands'] = 10000
    task_opt['light_unseen_test'] = task_opt['light_eval_unseen']

    random.seed(10)
    agent = RepeatLabelAgent(task_opt)
    world = create_task(task_opt, agent)

    # Populate dialogues from the LIGHT dataset
    samples = []
    curr_sample = []
    while True:
        world.parley()
        curr_sample.append(world.acts[0].copy())
        if world.acts[0]['episode_done']:
            if len(curr_sample) >= TRAIN_TURNS:
                samples.append(curr_sample)
            curr_sample = []
        if world.epoch_done():
            break

    train_samples = []
    task_opt['datatype'] = 'train:stream'
    task_opt['light_unseen_test'] = False
    agent = RepeatLabelAgent(task_opt)
    world = create_task(task_opt, agent)
    curr_sample = []
    while True:
        world.parley()
        curr_sample.append(world.acts[0].copy())
        if world.acts[0]['episode_done']:
            if len(curr_sample) >= TRAIN_TURNS:
                train_samples.append(curr_sample)
            curr_sample = []
        if world.epoch_done() or len(train_samples) > 2000:
            break

    # Set up temporary pools to pull tasks from
    use_train_samples = train_samples.copy()
    use_samples = train_samples.copy()

    # Set the task name to be the folder name
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    # append the contents of task_config.py to the configuration
    opt.update(task_config)

    # Select an agent_id that worker agents will be assigned in their world
    mturk_agent_roles = [LABEL_TYPE]

    opt['assignment_duration_in_seconds'] = 20 * 60

    # Instantiate an MTurkManager with the given options and a maximum number
    # of agents per world of 1 (based on the length of mturk_agent_ids)
    mturk_manager = MTurkManager(
        opt=opt,
        mturk_agent_ids=mturk_agent_roles,
        use_db=True,
    )
    mturk_manager.setup_server(
        task_directory_path=os.path.dirname(os.path.abspath(__file__)))

    # Create an onboard_function, which will be run for workers who have
    # accepted your task and must be completed before they are put in the
    # queue for a task world.
    completed_agents = []

    completed_train = {}

    def run_onboard(worker):
        nonlocal completed_agents
        if worker.worker_id in completed_agents:
            return
        else:
            world = LightEvalTestWorld(opt=opt, mturk_agent=worker)
            while not world.episode_done():
                world.parley()
            if world.did_complete:
                completed_agents.append(worker.worker_id)
            else:
                print(worker.worker_id, 'Failed the onboarding')
            world.shutdown()
            return world.prep_save_data([worker])

    mturk_manager.set_onboard_function(onboard_function=run_onboard)

    try:
        # Initialize run information
        mturk_manager.start_new_run()

        # Set up the sockets and threads to recieve workers
        mturk_manager.ready_to_accept_workers()

        # Create the hits as specified by command line arguments
        mturk_manager.create_hits(qualifications=[])

        # Check workers eligiblity acts as a filter, and should return
        # the list of all workers currently eligible to work on the task
        # Can be used to pair workers that meet certain criterea
        def check_workers_eligibility(workers):
            return workers

        eligibility_function = {
            'func': check_workers_eligibility,
            'multiple': True,
        }

        # Assign worker roles is used to determine what the role each worker
        # in the given worker list will play. Setting `id` to None will return
        # the worker to the pool rather than putting them in a given task,
        # which is useful for having tasks with different possible worker
        # counts.
        def assign_worker_roles(workers):
            workers[0].id = LABEL_TYPE

        # Define the task function, which will be run with workers that are
        # as the main task.
        global run_conversation

        def run_conversation(mturk_manager, opt, workers):
            nonlocal completed_train
            nonlocal use_samples
            nonlocal use_train_samples
            worker_id = workers[0].worker_id
            use_train = True
            if worker_id not in completed_train:
                completed_train[worker_id] = 0
            if completed_train[worker_id] >= TRAININGS:
                use_train = False

            # Create the real task world
            if not use_train:
                if len(use_samples) == 0:
                    # reset the pool if none are left
                    use_samples = samples.copy()
                sample = use_samples.pop()
            else:
                if len(use_train_samples) == 0:
                    # reset the pool if none are left
                    use_train_samples = train_samples.copy()
                sample = train_samples.pop()

            world = LightEvalTaskWorld(
                opt=opt,
                mturk_agents=workers,
                sample=sample,
                use_train=use_train,
                max_wrong=MAX_WRONG,
            )
            # run the world to completion
            while not world.episode_done():
                world.parley()

            # shutdown and review the work
            world.shutdown()
            world.review_work()

            if not world.completed and not use_train:
                samples.append(sample)
            if use_train and world.completed:
                completed_train[worker_id] += 1
                print('Worker passed train: ', worker_id)

            # Return the contents for saving
            return world.prep_save_data(workers)

        # Begin the task, allowing mturk_manager to start running the task
        # world on any workers who connect
        mturk_manager.start_task(
            eligibility_function=eligibility_function,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation
        )

    except BaseException:
        raise
    finally:
        print('Accepted agents:', repr(completed_agents))
        # Shutdown the manager and free all related resources
        mturk_manager.shutdown()


if __name__ == '__main__':
    main()

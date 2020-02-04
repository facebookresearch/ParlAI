#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List, Set
import json
import numpy as np
import os
from queue import Queue
import random
import time

from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld
from parlai.utils.misc import warn_once


DEFAULT_TASK_CONFIG = {
    'hit_title': 'Which Conversational Partner is Better?',
    'hit_description': 'Evaluate quality of conversations through comparison.',
    'hit_keywords': 'chat,evaluation,comparison,conversation',
}

task_queue: Queue = Queue()
onboarding_tasks: Dict[int, Dict] = {}
desired_tasks: Dict[int, Dict] = {}

workers_tasks_completed: Dict[int, List] = {}
workers_to_conversations_seen: Dict[int, List] = {}
workers_to_onboarding_tasks_todo: Dict[int, List] = {}
onboarding_failed_workers: Set = set()


def add_args(from_argv=False):
    """
    Add arguments to parser and either parse from commandline or initialize to defaults
    (for overriding in scripts)
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '--annotations-per-pair',
        type=int,
        default=1,
        help='Number of annotations per conversation comparison pair',
    )
    argparser.add_argument(
        '--pairings-filepath',
        type=str,
        default=None,
        help='path to the file containing the task dictionaries',
    )
    argparser.add_argument(
        '--task-config',
        type=dict,
        default=DEFAULT_TASK_CONFIG,
        help='dict with keys "hit_title", "hit_description", "hit_keywords", '
        'determining how task is displayed on MTurk site',
    )
    argparser.add_argument(
        '--s1-choice',
        type=str,
        default='I would prefer to talk to <Speaker 1>',
        help='text next to speaker 1 radio button',
    )
    argparser.add_argument(
        '--s2-choice',
        type=str,
        default='I would prefer to talk to <Speaker 2>',
        help='text next to speaker 2 radio button',
    )
    argparser.add_argument(
        '--question',
        type=str,
        default='Who would you prefer to talk to for a long conversation?',
        help='question to present to turker for comparison (e.g. "Which speaker is better?")',
    )
    argparser.add_argument(
        '--block-on-onboarding-fail',
        type=bool,
        default=True,
        help='whether to block on onboarding failure',
    )
    argparser.add_argument(
        '--subtasks-per-hit',
        type=int,
        default=5,
        help='number of subtasks/comparisons to do per hit',
    )
    argparser.add_argument(
        '--onboarding-threshold',
        type=float,
        default=0.75,
        help='minimum accuracy on onboarding tasks, as a float 0-1.0',
    )
    argparser.add_argument(
        '--seed', type=int, default=42, help='seed for random and np.random'
    )
    argparser.add_argument(
        '--softblock-list-path',
        type=str,
        default=None,
        help='Path to list of workers to softblock, separated by line breaks',
    )
    argparser.set_defaults(allowed_conversation=1)
    if from_argv:
        return argparser.parse_args()
    else:
        return argparser.parse_args(args=[])


def setup_task_queue(opt):
    """
    Initialize task queue to contain the specified number of instances of each pairing.
    """
    annotations_per_pair = opt['annotations_per_pair']
    internal_pair_id = 0

    ## Set up onboarding tasks
    if opt['pairings_filepath']:
        with open(opt['pairings_filepath'], 'r') as pf:
            for l in pf:
                pairing_dict = json.loads(l.strip())
                read_task_from_jsonl(
                    pairing_dict,
                    internal_pair_id,
                    opt['s1_choice'],
                    opt['s2_choice'],
                    opt['question'],
                )
                internal_pair_id += 1
    else:
        raise ValueError("You must provide a --pairings-filepath")

    ## Fill task queue
    for _i in range(annotations_per_pair):
        all_task_keys = list(desired_tasks.keys())
        np.random.shuffle(all_task_keys)
        for internal_pair_id in all_task_keys:
            task_queue.put(desired_tasks[internal_pair_id])
    # limit number of hits worker can do by default
    if opt['max_hits_per_worker'] == 0:
        opt['max_hits_per_worker'] = (
            len(desired_tasks) + len(onboarding_tasks)
        ) // opt['subtasks_per_hit']


def read_task_from_jsonl(
    pairing_dict, internal_pair_id, s1_choice, s2_choice, question
):
    """
    Build task dict according to expected format.
    """
    conv_order = random.choice([[0, 1], [1, 0]])
    task_data = {}
    specs = {}
    task_data['task_specs'] = specs
    task_data['pairing_dict'] = pairing_dict
    specs['conversation_order'] = conv_order
    specs['internal_pair_id'] = internal_pair_id
    specs['s1_choice'] = s1_choice
    specs['s2_choice'] = s2_choice
    specs['question'] = question
    if pairing_dict['is_onboarding']:
        specs['is_onboarding'] = True
        onboarding_tasks[internal_pair_id] = task_data
    else:
        desired_tasks[internal_pair_id] = task_data


def get_new_task_data(worker, tasks_per_hit):
    """
    Get next task for worker.

    Returns the next onboarding task if worker hasn't finished them all, or finds a task
    from the queue they haven't seen If they've seen everything in the queue, spin up an
    extra task (one that was in the queue and is now saturated)
    """
    worker_id = worker.worker_id
    task_data = get_onboarding_tasks(worker_id, tasks_per_hit)
    if len(task_data) == tasks_per_hit:
        return task_data
    tries = 0
    completed_tasks = workers_tasks_completed.get(worker_id, [])
    seen_conversations = workers_to_conversations_seen.get(worker_id, [])
    while (not task_queue.empty()) and tries < task_queue.qsize():
        try:
            next_task = task_queue.get()
        except Queue.Empty:
            break
        tries += 1

        internal_pair_id = next_task['task_specs']['internal_pair_id']
        dialogue0_id = next_task['pairing_dict']['dialogue_dicts'][0]['id']
        dialogue1_id = next_task['pairing_dict']['dialogue_dicts'][1]['id']

        if (  # make sure worker has not seen these conversations before
            internal_pair_id not in completed_tasks
            and dialogue0_id not in seen_conversations
            and dialogue1_id not in seen_conversations
        ):
            # track tasks and conversations seen
            completed_tasks.append(next_task['task_specs']['internal_pair_id'])
            workers_tasks_completed[worker_id] = completed_tasks
            seen_conversations.extend([dialogue0_id, dialogue1_id])
            workers_to_conversations_seen[worker_id] = seen_conversations
            task_data.append(next_task)
            if len(task_data) == tasks_per_hit:
                return task_data
        else:
            task_queue.put(next_task)
    # task queue containing num annotations requested of each pair is exhausted
    # b/c we released enough hits to guarantee reaching the requested num on average
    tasks_still_needed = tasks_per_hit - len(task_data)
    tasks_remaining = [id for id in desired_tasks.keys() if id not in completed_tasks]
    # get any pairings with conversations this worker has not seen to fill this hit
    tasks_chosen = [
        t
        for t in tasks_remaining
        if desired_tasks[t]['pairing_dict']['dialogue_dicts'][0]['id']
        not in seen_conversations
        and desired_tasks[t]['pairing_dict']['dialogue_dicts'][1]['id']
        not in seen_conversations
    ]
    if tasks_still_needed < len(tasks_chosen):
        tasks_chosen = np.random.choice(tasks_chosen, tasks_still_needed, replace=False)
    completed_tasks.extend(tasks_chosen)
    seen_conversations.extend(
        [
            desired_tasks[t]['pairing_dict']['dialogue_dicts'][0]['id']
            for t in tasks_chosen
        ]
    )
    seen_conversations.extend(
        [
            desired_tasks[t]['pairing_dict']['dialogue_dicts'][1]['id']
            for t in tasks_chosen
        ]
    )
    task_data.extend([desired_tasks[id] for id in tasks_chosen])
    return task_data


def return_task_data(worker_id, task_data):
    """
    When worker doesn't complete a task, return it to the queue or change their
    onboarding status depending on the task.
    """
    for subtask_data in task_data:
        if subtask_data['task_specs'].get('is_onboarding', False):
            workers_to_onboarding_tasks_todo[worker_id].append(
                subtask_data['task_specs']['internal_pair_id']
            )
        else:
            task_queue.put(subtask_data)
            try:
                workers_tasks_completed[worker_id].remove(
                    subtask_data['task_specs']['internal_pair_id']
                )
                workers_to_conversations_seen[worker_id].remove(
                    subtask_data['pairing_dict']['dialogue_dicts'][0]['id']
                )
                workers_to_conversations_seen[worker_id].remove(
                    subtask_data['pairing_dict']['dialogue_dicts'][1]['id']
                )
            except ValueError:
                print("WARNING: couldn't remove task from worker's history")


def get_onboarding_tasks(worker_id, tasks_per_hit):
    """
    Get the next onboarding task for this worker id.

    If the worker has never done a task, shuffle the onboarding tasks for them. If
    they've done all of the onboarding tasks or if there are no onboarding tasks, return
    None
    """
    if len(onboarding_tasks) == 0:
        return []
    onboarding_tasks_todo = workers_to_onboarding_tasks_todo.get(worker_id)
    if onboarding_tasks_todo is None:
        # new worker, instantiate their list of onboarding tasks
        onboarding_tasks_todo = list(onboarding_tasks.keys())
        np.random.shuffle(onboarding_tasks_todo)
        workers_to_onboarding_tasks_todo[worker_id] = onboarding_tasks_todo
    if len(onboarding_tasks_todo) == 0:
        # worker has completed all required onboarding tasks
        return []
    # get onboarding tasks for workers needing them
    num_tasks_to_return = min(len(onboarding_tasks_todo), tasks_per_hit)
    onboarding_tasks_chosen = onboarding_tasks_todo[:num_tasks_to_return]
    workers_to_onboarding_tasks_todo[worker_id] = onboarding_tasks_todo[
        num_tasks_to_return:
    ]
    return [onboarding_tasks[id] for id in onboarding_tasks_chosen]


def check_and_update_worker_approval(mturk_manager, worker_id, threshold, save_data):
    """
    Soft block workers who fail onboarding tasks, keep track of their status.
    """
    all_task_data = save_data['worker_data'][worker_id]['task_data']
    response_data = save_data['worker_data'][worker_id]['response']['task_data']
    num_onboarding_tasks = 0
    num_correct = 0

    for i in range(len(all_task_data)):
        is_onboarding = all_task_data[i]['pairing_dict'].get('is_onboarding', False)
        if not is_onboarding:
            # not an onboarding task, no need to check correctness
            continue
        worker_response = response_data[i]['speakerChoice']
        expected_response = all_task_data[i]['pairing_dict']['correct_answer']
        num_onboarding_tasks += 1
        if worker_response == expected_response:
            # count correct answers
            num_correct += 1
    if num_onboarding_tasks == 0:
        # no onboarding tasks found
        if worker_id in onboarding_failed_workers:
            # worker already failed onboarding, add pairings back to queue
            return_task_data(worker_id, all_task_data)
        return
    if (num_correct / num_onboarding_tasks) >= threshold:
        # worker did not fail onboarding
        return
    # worker failed onboarding, soft block and record
    mturk_manager.soft_block_worker(worker_id)
    onboarding_failed_workers.add(worker_id)


def main(opt):
    """
    Handles setting up and running a ParlAI-MTurk task by instantiating an MTurk manager
    and configuring it for the qa_data_collection task.
    """
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])

    # Set the task name to be the folder name
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    opt['task_description'] = {
        'num_subtasks': opt['subtasks_per_hit'],
        'question': opt['question'],
    }
    # append the contents of task_config.py to the configuration
    opt['frontend_version'] = 1
    opt.update(opt['task_config'])

    display_agent_name = 'RatingWorker'

    # Set up task queue before server
    setup_task_queue(opt)

    # Instantiate an MTurkManager with the given options and a maximum number
    # of agents per world of 1 (based on the length of mturk_agent_ids)
    mturk_manager = StaticMTurkManager(opt=opt)

    mturk_manager.setup_server(
        task_directory_path=os.path.dirname(os.path.abspath(__file__))
    )

    mturk_manager.set_onboard_function(onboard_function=None)

    try:
        # Initialize run information
        mturk_manager.start_new_run()

        task_group_id = mturk_manager.task_group_id
        if opt['block_on_onboarding_fail'] and opt['block_qualification'] is None:
            opt['block_qualification'] = task_group_id
            warn_once(
                "No block_qualification set in opt, automatically creating "
                "new qualification {}".format(task_group_id)
            )

        # Set up the sockets and threads to receive workers
        mturk_manager.ready_to_accept_workers()

        # Create the hits as specified by command line arguments
        mturk_manager.create_hits()

        def check_worker_eligibility(worker):
            return True

        def assign_worker_roles(workers):
            workers[0].id = display_agent_name

        def run_conversation(mturk_manager, opt, workers):
            task_data = get_new_task_data(workers[0], opt['subtasks_per_hit'])
            world = StaticMTurkTaskWorld(
                opt, mturk_agent=workers[0], task_data=task_data
            )
            while not world.episode_done():
                world.parley()

            world.shutdown()

            save_data = world.prep_save_data(workers)

            if not world.did_complete():
                return_task_data(workers[0].worker_id, task_data)
            elif opt['block_on_onboarding_fail']:
                # check whether workers failed onboarding
                check_and_update_worker_approval(
                    mturk_manager,
                    workers[0].worker_id,
                    opt['onboarding_threshold'],
                    save_data,
                )
            return save_data

        # Soft-block all chosen workers
        if not opt['is_sandbox'] and opt['softblock_list_path'] is not None:
            softblock_list = set()
            with open(opt['softblock_list_path'], 'r') as f:
                for line in f:
                    softblock_list.add(line.strip())
            print(f'Will softblock {len(softblock_list):d} workers.')
            for w in softblock_list:
                try:
                    print('Soft Blocking {}\n'.format(w))
                    mturk_manager.soft_block_worker(w)
                except Exception as e:
                    print(f'Did not soft block worker {w}: {e}')
                time.sleep(0.1)

        print("This run id: {}".format(task_group_id))

        # Begin the task, allowing mturk_manager to start running the task
        # world on any workers who connect
        mturk_manager.start_task(
            eligibility_function=check_worker_eligibility,
            assign_role_function=assign_worker_roles,
            task_function=run_conversation,
        )
    except BaseException:
        raise
    finally:
        # Any hits that aren't claimed or completed have to be shut down. Must
        # keep the world running until that point.
        mturk_manager.expire_all_unassigned_hits()
        # Shutdown the manager and free all related resources
        mturk_manager.shutdown()


if __name__ == '__main__':
    args = add_args(from_argv=True)
    main(args)

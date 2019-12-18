#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from queue import Queue
import os
import json
import numpy as np
import time

from parlai.core.params import ParlaiParser
from parlai.mturk.core.mturk_manager import StaticMTurkManager
from parlai.mturk.core.worlds import StaticMTurkTaskWorld
from parlai.mturk.tasks.acute_eval.task_config import task_config
import parlai.mturk.core.mturk_utils as mturk_utils
from parlai.utils.misc import warn_once


task_queue = Queue()
onboarding_tasks = {}
onboarding_conv_ids = []
desired_tasks = {}
conversations_to_tasks = {}

workers_tasks_completed = {}
workers_to_conversations_seen = {}
workers_to_onboarding_tasks_todo = {}
onboarding_failed_workers = []


def add_args(from_argv=False):
    """ Add arguments to parser and either parse from commandline or initialize
    to defaults (for overriding in scripts)
    """
    argparser = ParlaiParser(False, False)
    argparser.add_parlai_data_path()
    argparser.add_mturk_args()
    argparser.add_argument(
        '--dialogs_path',
        type=str,
        default=None,
        help='path to folder with conversation log files for evaluation',
    )
    argparser.add_argument(
        '--annotations_per_pair',
        type=int,
        default=1,
        help='Number of annotations per conversation comparison pair',
    )
    argparser.add_argument(
        '--pair_data',
        type=list,
        default=None,
        help='list of (conv1, conv2, hit, desc) (for pre-chosen pairs, e.g. for replicating previous experiments)',
    )
    argparser.add_argument(
        '--s1_choice',
        type=str,
        default='I would prefer to talk to <Speaker 1>',
        help='text next to speaker 1 radio button',
    )
    argparser.add_argument(
        '--s2_choice',
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
        '--correctness_is_flipped',
        default=False,
        action='store_true',
        help='question phrasing flips the better model - e.g. question is "Which speaker is more boring?"',
    )
    argparser.add_argument(
        '--model_comparisons',
        type=str,
        help='list of model pairs to compare, comma separated. E.g. ["transformer,human_eval"] ',
    )
    argparser.add_argument(
        '--pairs_per_matchup',
        type=int,
        default=160,
        help='Number of conversation pairs to generate for the comparison',
    )
    argparser.add_argument(
        '--block_on_onboarding_fail',
        type=bool,
        default=True,
        help='whether to block on onboarding failure',
    )
    argparser.add_argument(
        '--onboarding_tasks',
        type=list,
        default=None,
        help='onboarding tasks to screen workers with, list of (wrong_convid, right_convid, matchup) tuples',
    )
    argparser.add_argument(
        '--subtasks_per_hit',
        type=int,
        default=5,
        help='number of subtasks/comparisons to do per hit',
    )
    argparser.add_argument(
        '--onboarding_threshold',
        type=float,
        default=0.75,
        help='minimum accuracy on onboarding tasks, as a float 0-1.0',
    )
    argparser.add_argument('--seed', type=int, default=42, help='np.random seed')
    argparser.set_defaults(allowed_conversation=1)
    if from_argv:
        return argparser.parse_args()
    else:
        return argparser.parse_args(args=[])


def list_files(folder):
    for fn in os.listdir(folder):
        full_fn = os.path.join(folder, fn)
        if os.path.isfile(full_fn):
            yield full_fn
        elif os.path.isdir(full_fn):
            for sfn in list_files(full_fn):
                yield sfn


def setup_task_queue(opt):
    """ Initialize task queue to contain the specified number of instances of
    each pairing
    """
    # hacky fix for the parlai parser hacky fix
    data_folder = opt['dialogs_path'].replace('-', '_')
    annotations_per_pair = opt['annotations_per_pair']
    all_conv_data = {}
    conv_ids_by_model = {}
    internal_id = 0

    # read in all conversation data
    for data_fn in os.listdir(data_folder):
        full_data_fn = os.path.join(data_folder, data_fn)
        with open(full_data_fn, 'r') as dialog_data_file:
            model_name = data_fn.split('.')[0]
            model_convs = []
            conv_ids_by_model[model_name] = model_convs

            for l in dialog_data_file:
                try:
                    single_task_json = json.loads(l)
                except:
                    print("ILLEGAL FORMATTING:", l)
                    raise RuntimeError('One or more files are not valid .jsonl')
                id = single_task_json.get('conversation_id')
                all_conv_data[id] = single_task_json
                model_convs.append(id)

    ## Set up onboarding tasks
    if opt['onboarding_tasks']:
        for (id1, id2, matchup) in opt['onboarding_tasks']:
            make_task_from_ids(
                id1,
                id2,
                internal_id,
                all_conv_data,
                opt['s1_choice'],
                opt['s2_choice'],
                opt['question'],
                opt['correctness_is_flipped'],
                matchup=matchup,
                is_qual=True,
            )
            internal_id += 1
    else:
        warn_once("No onboarding task provided")

    ## Create main tasks from list of tuples
    if opt['pair_data']:
        print('Creating {} tasks from --pair_data'.format(len(opt['pair_data'])))
        for (id1, id2, hit_id, matchup) in opt['pair_data']:
            make_task_from_ids(
                id1,
                id2,
                internal_id,
                all_conv_data,
                opt['s1_choice'],
                opt['s2_choice'],
                opt['question'],
                opt['correctness_is_flipped'],
                hit_id,
                matchup,
            )
            internal_id += 1
    # make desired tasks randomly from scratch
    elif opt['model_comparisons']:
        print('Creating random tasks for model pairs in --model_comparisons')
        for model_0, model_1 in opt['model_comparisons']:
            if model_0 not in conv_ids_by_model:
                raise Error("{} is not a valid model name".format(model0))
            if model_1 not in conv_ids_by_model:
                raise Error("{} is not a valid model name".format(model1))
            num_pairs = opt['pairs_per_matchup']
            matchup_name = '{},{}'.format(model_0, model_1)
            conv_pairs = []
            all_model1_convs = [
                id for id in conv_ids_by_model[model_0] if id not in onboarding_conv_ids
            ]
            all_model2_convs = [
                id for id in conv_ids_by_model[model_1] if id not in onboarding_conv_ids
            ]
            while len(conv_pairs) < num_pairs:
                id1 = np.random.choice(all_model1_convs)
                id2 = np.random.choice(all_model2_convs)
                if (id1, id2) in conv_pairs:
                    continue
                conv_pairs.append((id1, id2))

                make_task_from_ids(
                    id1,
                    id2,
                    internal_id,
                    all_conv_data,
                    opt['s1_choice'],
                    opt['s2_choice'],
                    opt['question'],
                    opt['correctness_is_flipped'],
                    matchup=matchup_name,
                )
                internal_id += 1
    else:
        raise NotImplementedError("Provide --pair_data or --model_comparison")

    ## Fill task queue
    for i in range(annotations_per_pair):
        all_task_keys = list(desired_tasks.keys())
        np.random.shuffle(all_task_keys)
        for internal_id in all_task_keys:
            task_queue.put(desired_tasks[internal_id])
    # limit number of hits worker can do by default
    if opt['max_hits_per_worker'] == 0:
        opt['max_hits_per_worker'] = (len(desired_tasks) + len(onboarding_tasks)) // opt[
            'comparisons_per_hit'
        ]


def make_task_from_ids(
    id1,
    id2,
    internal_id,
    all_conv_data,
    s1_choice,
    s2_choice,
    question,
    is_flipped,
    hitid='',
    matchup='',
    is_qual=False,
):
    """ Build task dict according to expected format
    """
    conv_orders = [[0, 1], [1, 0]]
    conv1 = all_conv_data.get(id1)
    conv2 = all_conv_data.get(id2)
    if conv1 is None or conv2 is None:
        raise Exception("One of assignment ids {}, {} not found".format(id1, id2))
    task_data = {}
    task_data['conversations'] = [conv1, conv2]
    specs = {}
    task_data['task_specs'] = specs
    specs['comparison_type'] = matchup
    specs['original_hit_id'] = hitid
    specs['conversation_order'] = conv_orders[np.random.choice([0, 1])]
    specs['internal_id'] = internal_id
    specs['s1_choice'] = s1_choice
    specs['s2_choice'] = s2_choice
    specs['question'] = question
    specs['correctness_is_flipped'] = is_flipped
    specs['speakers_to_eval'] = ['model', 'model']
    if is_qual:
        specs['is_onboarding'] = True
        onboarding_conv_ids.extend([conv1, conv2])
        onboarding_tasks[internal_id] = task_data
    else:
        desired_tasks[internal_id] = task_data
        for id in [id1, id2]:
            if id not in conversations_to_tasks:
                conversations_to_tasks[id] = []
            conversation_task_list = conversations_to_tasks[id]
            conversation_task_list.append(id)


def get_new_task_data(worker, tasks_per_hit):
    """ Get next task for worker. Returns the next onboarding task if worker
    hasn't finished them all, or finds a task from the queue they haven't seen
    If they've seen everything in the queue, spin up an extra task (one that
    was in the queue and is now saturated)
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
        if (  # make sure worker has not seen these conversations before
            next_task['task_specs']['internal_id'] not in completed_tasks
            and next_task['conversations'][0] not in seen_conversations
            and next_task['conversations'][1] not in seen_conversations
        ):
            completed_tasks.append(next_task['task_specs']['internal_id'])
            workers_tasks_completed[worker_id] = completed_tasks
            seen_conversations.extend(next_task['conversations'])
            workers_to_conversations_seen[worker_id] = seen_conversations
            task_data.append(next_task)
            if len(task_data) == tasks_per_hit:
                return task_data
        else:
            task_queue.put(next_task)
    # task queue is exhausted
    tasks_still_needed = tasks_per_hit - len(task_data)
    tasks_remaining = [id for id in desired_tasks.keys() if id not in completed_tasks]
    tasks_chosen = [
        t
        for t in tasks_remaining
        if desired_tasks[t]['conversations'][0] not in seen_conversations
        and desired_tasks[t]['conversations'][1] not in seen_conversations
    ]
    if tasks_still_needed < len(tasks_chosen):
        tasks_chosen = np.random.choice(tasks_chosen, tasks_still_needed, replace=False)
    completed_tasks.extend(tasks_chosen)
    seen_conversations.extend(
        [desired_tasks[t]['conversations'][0] for t in tasks_chosen]
    )
    seen_conversations.extend(
        [desired_tasks[t]['conversations'][1] for t in tasks_chosen]
    )
    task_data.extend([desired_tasks[id] for id in tasks_chosen])
    return task_data


def return_task_data(worker_id, task_data):
    """ When worker doesn't complete a task, return it to the queue or
    change their onboarding status depending on the task"""
    for subtask_data in task_data:
        if subtask_data['task_specs'].get('is_onboarding', False):
            workers_to_onboarding_tasks_todo[worker_id].append(
                subtask_data['task_specs']['internal_id']
            )
        else:
            task_queue.put(subtask_data)
            try:
                workers_tasks_completed[worker_id].remove(
                    subtask_data['task_specs']['internal_id']
                )
                workers_to_conversations_seen[worker_id].remove(
                    subtask_data['conversations'][0]
                )
                workers_to_conversations_seen[worker_id].remove(
                    subtask_data['conversations'][1]
                )
            except ValueError():
                print("WARNING: couldn't remove task from worker's history")


def get_onboarding_tasks(worker_id, tasks_per_hit):
    """ Get the next onboarding task for this worker id. If the worker has never
    done a task, shuffle the onboarding tasks for them. If they've done all
    of the onboarding tasks or if there are no onboarding tasks, return None
    """
    if len(onboarding_tasks) == 0:
        return []
    onboarding_tasks_todo = workers_to_onboarding_tasks_todo.get(worker_id)
    if onboarding_tasks_todo is None:
        onboarding_tasks_todo = list(onboarding_tasks.keys())
        np.random.shuffle(onboarding_tasks_todo)
        workers_to_onboarding_tasks_todo[worker_id] = onboarding_tasks_todo
    if len(onboarding_tasks_todo) == 0:
        return []
    num_tasks_to_return = min(len(onboarding_tasks_todo), tasks_per_hit)
    onboarding_tasks_chosen = onboarding_tasks_todo[:num_tasks_to_return]
    workers_to_onboarding_tasks_todo[worker_id] = onboarding_tasks_todo[
        num_tasks_to_return:
    ]
    return [onboarding_tasks[id] for id in onboarding_tasks_chosen]


def check_and_update_worker_approval(mturk_manager, worker_id, threshold, save_data):
    """ Soft block workers who fail onboarding tasks, keep track of their status
    """
    task_data = save_data['worker_data'][worker_id]['task_data']
    response_data = save_data['worker_data'][worker_id]['response']['task_data']
    num_onboarding_tasks = 0
    num_correct = 0
    for i in range(len(task_data)):
        task_specs = task_data[i]['task_specs']
        if not task_specs.get('is_onboarding', False):
            continue
        worker_response = float(response_data[i]['speakerChoice'])
        expected_response = (
            1
            if (
                (
                    task_specs['conversation_order'] == [1, 0]
                    and not task_specs['correctness_is_flipped']
                )
                or (
                    task_specs['conversation_order'] == [0, 1]
                    and task_specs['correctness_is_flipped']
                )
            )
            else 2
        )
        num_onboarding_tasks += 1
        if worker_response == expected_response:
            num_correct += 1
    if num_onboarding_tasks == 0:
        if worker_id in onboarding_failed_workers:
            return_task_data(worker_id, task_data)
        return
    if (num_correct / num_onboarding_tasks) >= threshold:
        return
    mturk_manager.soft_block_worker(worker_id)
    onboarding_failed_workers.append(worker_id)


def main(opt):
    """Handles setting up and running a ParlAI-MTurk task by instantiating
    an MTurk manager and configuring it for the qa_data_collection task
    """
    np.random.seed(opt['seed'])

    # Set the task name to be the folder name
    opt['task'] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    # append the contents of task_config.py to the configuration
    opt.update(task_config)

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

    if opt['block_on_onboarding_fail'] and opt['block_qualification'] is None:
        raise Exception(
            "Set block_qualification or set block_on_onboarding_fail to False"
        )

    try:
        # Initialize run information
        mturk_manager.start_new_run()
        # Set up the sockets and threads to recieve workers
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
                check_and_update_worker_approval(
                    mturk_manager,
                    workers[0].worker_id,
                    opt['onboarding_threshold'],
                    save_data,
                )
            return save_data

        print("This run id: {}".format(mturk_manager.task_group_id))

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

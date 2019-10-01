#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Generates a flattened version of a ParlAI task, i.e., for tasks with
multi-turn dialogues (episodes), this will generate a task with single
example episodes, in which the context from previous dialogue turns in each
episode are included in the 'text' field of each example.

In order to use this teacher, specify the task flag as follows:
`--task flattened:task:<ORIGINAL TASK NAME>`.

As an example, try running:

`python examples/display_data.py -t flattened:task:convai2`
"""

from parlai.core.agents import get_task_module
from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher

from collections import deque
from copy import deepcopy
import json
import os
import random
from tqdm import tqdm


def flatten(episode, context_length, include_labels=True, delimiter='\n'):
    context = deque(maxlen=context_length if context_length > 0 else None)
    new_episode = []

    for ex in episode:
        context.append(ex.get('text', ''))
        # add context
        if len(context) > 1:
            ex.force_set('text', delimiter.join(context))
        # set episode_done to be True
        ex.force_set('episode_done', True)
        labels = ex.get('labels', ex.get('eval_labels', None))
        if labels is not None and include_labels:
            context.append(random.choice(labels))

        new_episode.append(ex)

    return new_episode


def get_original_task_module(opt, multi_possible=False):
    modules = []
    tasks = opt['task'].split(',')
    if not multi_possible:
        assert len(tasks) == 1

    for task in tasks:
        if len(task.split(':')) < 3:
            raise RuntimeError(
                '\n\n********************************************************\n'
                'Must specify original task using the following format:\n'
                '`--task flattened:task:<ORIGINAL TASK NAME>`'
                '\n********************************************************\n'
            )
        original_task = ':'.join(task.split(':')[2:])
        task_module = get_task_module(original_task)
        modules.append(task_module)

    if multi_possible:
        return modules

    return modules[0]


class TaskTeacher(FixedDialogTeacher):
    """
    Generates a flattened version of a ParlAI task, i.e., for tasks with
    multi-turn dialogues (episodes), this will generate a task with single
    example episodes, in which the context from previous dialogue turns in each
    episode are included in the 'text' field of each example.
    """

    @staticmethod
    def add_cmdline_args(parser):
        agent = parser.add_argument_group('Flattened Teacher Args')
        agent.add_argument(
            '--flatten-include-labels',
            type='bool',
            default=True,
            help='Include labels in the history when flattening an episode',
        )
        agent.add_argument(
            '--flatten-delimiter',
            type=str,
            default='\n',
            help='How to join the dialogue history from previous turns.',
        )
        agent.add_argument(
            '--max-context-length', type=int, default=-1, help='Maximum context length'
        )
        agent.add_argument(
            '--invalidate-cache',
            type='bool',
            default=False,
            help='Set this to True to rebuild the data (may want to do this if '
            'original data has changed or you want to rebuild with new options)',
        )

        # Add the arguments for the teacher
        opt = parser.parse_and_process_known_args()[0]
        tasks = get_original_task_module(opt, multi_possible=True)
        for task in tasks:
            if hasattr(task, 'add_cmdline_args'):
                task.add_cmdline_args(parser)

        return parser

    def __init__(self, opt, shared=None):
        self.opt = opt

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self.data = self._setup_data(opt)

        super().__init__(opt, shared)
        self.reset()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def _setup_data(self, opt):
        # possibly make new data directory
        original_task_module = get_original_task_module(opt)
        self.original_task_name = ':'.join(opt['task'].split(':')[2:])
        teacher_name = self.original_task_name.replace(':', '_') + '_flattened'
        self.save_dir = os.path.join(opt['datapath'], teacher_name)
        os.makedirs(self.save_dir, exist_ok=True)

        datatype = opt['datatype'].split(':')[0]
        self.save_path = os.path.join(self.save_dir, f'{datatype}_data.json')

        data = self.load_data(opt)
        if data is not None:
            # successfully load data
            return data

        # build the original teacher
        teacher_opt = deepcopy(opt)
        teacher_opt['task'] = self.original_task_name
        teacher = original_task_module(teacher_opt)

        # did not load data, let's build it!
        total_exs = teacher.num_examples()
        num_exs = 0

        context_length = opt['max_context_length']
        include_labels = opt['flatten_include_labels']
        delimiter = opt['flatten_delimiter']

        total_eps = teacher.num_episodes()
        progress_bar = tqdm(
            total=total_eps, unit='ex', unit_scale=True, desc='Building flattened data'
        )

        all_episodes = []
        while num_exs < total_exs:
            current_episode = []
            episode_done = False

            while not episode_done:
                # TODO: eventually all teachers should return Messages, so
                # we should assert this
                action = Message(teacher.act())
                current_episode.append(action)
                episode_done = action.get('episode_done', False)
                num_exs += 1

            # flatten the episode into 1-example episodes with context
            flattened_ep = flatten(
                current_episode,
                context_length,
                include_labels=include_labels,
                delimiter=delimiter,
            )
            all_episodes += flattened_ep

            progress_bar.update(1)

        # save data for future use
        self.save_data(all_episodes)

        return all_episodes

    def load_data(self, opt):
        if not os.path.exists(self.save_path):
            # data has not been built yet
            return None

        if opt['invalidate_cache']:
            # invalidate the cache and remove the existing data
            print(' [ WARNING: invalidating cache and rebuilding the data. ]')
            os.remove(self.save_path)
            return None

        print(f' [ Data already exists. Loading from: {self.save_path} ]')
        with open(self.save_path, 'rb') as f:
            data = json.load(f)

        return data

    def save_data(self, data):
        json_data = json.dumps(data)
        with open(self.save_path, 'w') as f:
            f.write(json_data)

        print(f'[ Data successfully saved to path: {self.save_path} ]')
        return

    def get(self, episode_idx, entry_idx=0):
        return Message(self.data[episode_idx])

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class DefaultTeacher(TaskTeacher):
    pass

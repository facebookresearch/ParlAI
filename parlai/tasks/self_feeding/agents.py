#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import json
import os
import re

import numpy as np

from parlai.core.teachers import ParlAIDialogTeacher, MultiTaskTeacher
from projects.self_feeding.utils import add_person_tokens
from parlai.utils.io import PathManager
from .build import build


def _path(opt, filename, add_suffix=False):
    build(opt)
    if add_suffix:
        subtask = opt['subtask']
        if subtask == 'dialog':
            suffix = 'hh'
        elif subtask == 'satisfaction':
            suffix = 'st'
        elif subtask == 'feedback':
            suffix = 'fb'
        else:
            raise ValueError(f"Unrecognized subtask: {subtask}")
        filename += f"_{suffix}.txt"
    dp = os.path.join(opt['datapath'], 'self_feeding')
    return os.path.join(dp, filename)


class SelfFeedingTeacher(ParlAIDialogTeacher):
    """
    Teacher for the SelfFeedingAgent.

    opt['datatype'] determines whether we use the designated filepath ('train') or one
    of the eval files ('valid', 'test'), which are identical regardless of     what
    training set is being used.
    """

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if 'subtask' not in opt:
            print(
                'Warning: SelfFeedingteacher should be assigned subtask. '
                'Defaulting to dialog'
            )
            opt['subtask'] = 'dialog'

        # Use 'in' to also capture 'train:ordered:stream'
        if 'train' in opt['datatype']:
            # Use the filename explicitly given with the flag if available
            # Otherwise, use train_xx.txt where xx is inferred from the subtask
            train_file_flag = f"{opt['subtask'][:3]}_train"
            if opt.get(train_file_flag, None):
                path = _path(opt, opt[train_file_flag], add_suffix=False)
            else:
                path = _path(opt, "train", add_suffix=True)
        else:
            # Use the filename explicitly given with the flag if available
            # Otherwise, use the datatype (valid_xx.txt or test_xx.txt) where xx is
            # inferred from the subtask.
            eval_file_flag = f"{opt['subtask'][:3]}_{opt['datatype']}"
            if opt.get(eval_file_flag, None):
                path = _path(opt, opt[eval_file_flag], add_suffix=False)
            else:
                path = _path(opt, opt['datatype'].split(':')[0], add_suffix=True)

        if not PathManager.exists(path):
            raise ValueError("Unrecognized filepath: {}".format(path))

        opt['parlaidialogteacher_datafile'] = path
        opt['datafile'] = path
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        project = argparser.add_argument_group('Self-Feeding Tasks')
        project.add_argument(
            '-st',
            '--subtasks',
            type=str,
            help='comma-separated list of tasks used by MTL teacher',
        )
        project.add_argument(
            '-dia-train',
            '--dia-train',
            type=str,
            help='the filename to train on for the dialog task',
        )
        project.add_argument(
            '-fee-train',
            '--fee-train',
            type=str,
            help='the filename to train on for the feedback task',
        )
        project.add_argument(
            '-sat-train',
            '--sat-train',
            type=str,
            help='the filename to train on for the satisfaction task',
        )
        project.add_argument(
            '-dia-valid',
            '--dia-valid',
            type=str,
            help='the filename to eval on for the dialog task',
        )
        project.add_argument(
            '-fee-valid',
            '--fee-valid',
            type=str,
            help='the filename to eval on for the feedback task',
        )
        project.add_argument(
            '-sat-valid',
            '--sat-valid',
            type=str,
            help='the filename to eval on for the satisfaction task',
        )
        project.add_argument(
            '-dia-test',
            '--dia-test',
            type=str,
            help='the filename to eval on for the dialog task',
        )
        project.add_argument(
            '-fee-test',
            '--fee-test',
            type=str,
            help='the filename to eval on for the feedback task',
        )
        project.add_argument(
            '-sat-test',
            '--sat-test',
            type=str,
            help='the filename to eval on for the satisfaction task',
        )
        project.add_argument(
            '-trial',
            '--trial',
            type=int,
            default=0,
            help='the index of a repeated trial (not used in code)',
        )
        project.add_argument(
            '-mt',
            '--max-train',
            type=int,
            default=0,
            help='if non-zero, only the first max-train examples will '
            'be used if it is read by an instance of '
            'ParlaiDialogTeacher',
        )
        argparser.set_defaults(history_size=2)

    def _setup_data(self, path):  # Make private method for ParlAIDialogTeacher
        """
        Reads data in the fbdialog format.

        Returns ``((x,y,r,c), new_episode?)`` tuples.
        """
        print("[ Loading Self-Feeding text data:" + path + "]")
        self.episodes = []
        self.num_exs = 0
        self.max_train = self.opt.get('max_train', 0)
        with PathManager.open(path, 'r') as f:
            for line in f.readlines():
                if self.max_train and self.num_exs >= self.max_train:
                    break
                parley = json.loads(line)

                # NOTE: History is trimmed here, not by TorchAgent (except in
                # interactive mode)
                if self.opt['history_size'] == 0:
                    parley['context'] = '__null__'
                elif self.opt['history_size'] > 0:
                    utterances = re.split(r'__p\d__', parley['context'])[1:]
                    trimmed = utterances[-(self.opt['history_size']) :]
                    parley['context'] = add_person_tokens(trimmed, last_speaker=1)

                # WARNING: STRIPPING AWAY MEMORIES
                parley['memories'] = []

                episode = {
                    'text': parley['context'],
                    'labels': [parley['response']],
                    'label_candidates': parley.get('candidates', []),
                    'reward': parley.get('reward', 0),
                    'episode_done': True,
                }

                # Convert integer labels (e.g., polarization dataset) to strings
                episode['labels'] = [str(l) for l in episode['labels']]

                self.num_exs += 1
                self.episodes.append([episode])


class SelfFeedingMTLTeacher(MultiTaskTeacher):
    """
    Creates a teacher that is actually a set of teachers each based on a task string--
    each of these teachers will get called in turn, either randomly or in order. They
    are all in the same world (they are the same agent switching tasks).

    More specifically, this child class of MultiTaskTeacher supports multitask learning
    with batches (ensuring that all batches only have data from a single task at a time)
    """

    def __init__(self, opt, shared=None):
        if opt['task'] == 'self_feeding:SelfFeedingMTLTeacher':
            # This does not happen except in tests such as test_new_tasks.py
            opt = copy.deepcopy(opt)
            opt['task'] = 'self_feeding:SelfFeedingTeacher'
        super().__init__(opt, shared)

        # TODO: allow user to specify other strategies for creating sampling_prob
        # uniform, batch_proportion, etc.
        num_batches = np.array([t.num_examples() / t.bsz for t in self.tasks])
        self.sampling_prob = num_batches / np.sum(num_batches)

        self.task_idx_assignment = -1  # This will be updated by BatchWorld
        self.new_task = True
        self.random = opt.get('datatype') == 'train'

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)

    def observe(self, observation):
        return self.tasks[self.task_idx].observe(observation)

    def batch_act(self, batch_observation):
        # We use only one task for each batch
        task_idx = self.get_task_index()
        actions = []
        for _ in range(self.tasks[task_idx].bsz):
            action = self.tasks[task_idx].act()
            # Pass the name of the task currently being worked on
            action['subtask'] = self.tasks[task_idx].opt['subtask']
            actions.append(action)
        return actions

    def act(self):
        self.task_idx = self.get_task_index()
        if self.task_idx < 0:
            return {'episode_done': True}
        action = self.tasks[self.task_idx].act()
        # Pass the name of the task currently being worked on
        action['subtask'] = self.tasks[self.task_idx].opt['subtask']
        return action

    def get_task_index(self):
        if self.opt['datatype'] == 'train':
            return np.random.choice(range(len(self.tasks)), p=self.sampling_prob)
        else:
            for i, subtask in enumerate(self.tasks):
                if not subtask.epoch_done():
                    return i
        # If this is reached, all tasks are done, so return sentinel
        return -1


class DialogTeacher(SelfFeedingTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtask'] = 'dialog'
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)


class FeedbackTeacher(SelfFeedingTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtask'] = 'feedback'
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)


class SatisfactionTeacher(SelfFeedingTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtask'] = 'satisfaction'
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)


class DiafeeTeacher(SelfFeedingMTLTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtasks'] = ['dialog', 'feedback']
        # Expand abbreviated task name ('both') into full task names
        train_files = [opt['dia_train'], opt['fee_train']]
        assert len(opt['subtasks']) == len(train_files)
        tasks = [
            f'self_feeding:{subtask}:{train_file}'
            for subtask, train_file in zip(opt['subtasks'], train_files)
        ]
        opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)


class DiasatTeacher(SelfFeedingMTLTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtasks'] = ['dialog', 'satisfaction']
        # Expand abbreviated task name ('both') into full task names
        train_files = [opt['dia_train'], opt['sat_train']]
        assert len(opt['subtasks']) == len(train_files)
        tasks = [
            f'self_feeding:{subtask}:{train_file}'
            for subtask, train_file in zip(opt['subtasks'], train_files)
        ]
        opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)


class AllTeacher(SelfFeedingMTLTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtasks'] = ['dialog', 'feedback', 'satisfaction']
        # Expand abbreviated task name ('all') into full task names
        train_files = [opt['dia_train'], opt['fee_train'], opt['sat_train']]
        assert len(opt['subtasks']) == len(train_files)
        tasks = [
            f'self_feeding:{subtask}:{train_file}'
            for subtask, train_file in zip(opt['subtasks'], train_files)
        ]
        opt['task'] = ','.join(tasks)
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dialog Blender Dataset implementation for ParlAI. 
"""

import os
import json
from parlai.core.opt import Opt
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.teachers import DialogTeacher
from parlai.utils.misc import warn_once
from parlai.core.message import Message
from parlai.core.metrics import AverageMetric, BleuMetric
from parlai.utils.io import PathManager
from parlai.core.worlds import create_task

from parlai.tasks.dialog_blender.blender import Blender

MAX_TRAIN_DIALOGS = int(1e5)

DOMAINS = [
    'flights',
    'food-ordering',
    'hotels',
    'movies',
    'restaurant-search',
    'sports',
    'music',
]


class DialogBlender(DialogTeacher):
    """
    Teacher which produces both API calls and NLG responses.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument('--include-ontology', type=bool, default=False)
        argparser.add_argument(
            '--domains',
            nargs='+',
            default=DOMAINS,
            choices=DOMAINS,
            help='Uses last passed in configuration.',
        )
        argparser.add_argument(
            "--tasks_to_blend",
            type=str,
            default="google_sgd,dailydialog",
        )
        argparser.add_argument(
            "--blend_mode",
            type=str,
            default="fixed_interleave",
            choices=["concat", "random_interleave", "fixed_interleave"],
        )
        argparser.add_argument(
            "--eval_task",
            type=str,
            default="default",
        )
        return argparser

    def __init__(self, opt: Opt, shared=None):
        self.opt = opt
        self.fold = opt['datatype'].split(':')[0]
        opt['datafile'] = self.fold
        self.blender = Blender(opt["blend_mode"])
        self.tasks_to_blend = opt['tasks_to_blend'].split(',')
        self.eval_task = opt["eval_task"]
        if self.eval_task == "default":
            self.eval_task = self.tasks_to_blend[0]
        else:
            assert self.eval_task in self.tasks_to_blend
        self.dpath = os.path.join(opt['datapath'], 'dialog_blender')
        super().__init__(opt, shared)

    def _get_world(self, task_id):
        curr_task_opt = self.opt.copy()
        curr_task_opt['task'] = task_id
        curr_task_opt['datapath'] = os.path.join(self.dpath)
        agent = RepeatLabelAgent(curr_task_opt)
        world = create_task(curr_task_opt, agent)
        return world

    def _get_dialog(self, world):
        turns = []
        world.parley()
        turns.append(world.get_acts()[0])
        while not world.episode_done():
            world.parley()
            turns.append(world.get_acts()[0])
        return turns

    def _merge_dialogs(self, dialogs_all_tasks):
        merged_dialogs = []
        for dialog_id, dialogs in enumerate(zip(*dialogs_all_tasks)):
            merged_dialogs.append(self.blender.blend(dialogs))
        return merged_dialogs

    def _dialogs_split(self, fold):
        if fold != "train":
            eval_world = self._get_world(self.eval_task)
            return [self._get_dialog(eval_world) for _ in eval_world.num_episodes()]

        dialogs_all_tasks = []
        for task_id in self.tasks_to_blend:
            world = self._get_world(task_id)
            dialogs_all_tasks.append([self._get_dialog(world) for _ in range(MAX_TRAIN_DIALOGS)])
        return self._merge_dialogs(dialogs_all_tasks)

    def setup_data(self, fold):
        for dialog in self._dialogs_split(fold):
            for turn_id, turn in enumerate(dialog):
                is_first_turn = turn_id == 0
                turn['label'] = turn['labels'][0]
                del turn['labels']
                if 'label_candidates' in turn:
                    del turn['label_candidates']
                del turn['episode_done']
                yield turn, is_first_turn


class DefaultTeacher(DialogBlender):
    pass

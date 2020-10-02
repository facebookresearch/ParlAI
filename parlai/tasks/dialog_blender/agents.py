#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Google The Schema-Guided Dialogue(SGD) Dataset implementation for ParlAI.
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


class DialogBlender(DialogTeacher):
    """
    Teacher which produces both API calls and NLG responses.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser.add_argument(
            "--dialogs_to_blend",
            type=str,
            default="google_sgd,dailydialog",
        )
        return argparser


    def __init__(self, opt: Opt, shared=None):
        self.opt = opt
        self.fold = opt['datatype'].split(':')[0]
        opt['datafile'] = self.fold
        self.tasks_to_blend = opt['dialogs_to_blend'].split(',')
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
        while not world.episode_done():
            world.parley()
            turns.append(world.get_acts()[0])
        return turns

    def _merge_dialogs(self, dialogs):
        merged_dialog = []
        for task_id in dialogs:
            merged_dialog += dialogs[task_id]
        return merged_dialog
        
    def setup_data(self, fold):
        worlds = {}
        for task_id in self.tasks_to_blend:
            worlds[task_id] = self._get_world(task_id)

        num_dialogs = 10000
        for _ in range(num_dialogs):
            dialog = {}
            for task_id in self.tasks_to_blend:
                dialog[task_id] = self._get_dialog(worlds[task_id])
            for turn_id, turn in enumerate(self._merge_dialogs(dialog)):
                is_first_turn = turn_id == 0
                turn['label'] = turn['labels'][0]
                del turn['labels']
                del turn['episode_done']
                yield turn, is_first_turn


class DefaultTeacher(DialogBlender):
    pass

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Teachers for the MovieDialog task.

From Dodge et al. '15. Link: https://arxiv.org/abs/1511.06931

Task 1: Closed-domain QA dataset asking templated questions about movies,
answerable from Wikipedia.

Task 2: Questions asking for movie recommendations.

Task 3: Dialogs discussing questions about movies as well as recommendations.

Task 4: Dialogs discussing Movies from Reddit (the /r/movies SubReddit).
"""
from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build

import copy
import os

tasks = {}
tasks[1] = os.path.join('task1_qa', 'task1_qa_pipe_')
tasks[2] = os.path.join('task2_recs', 'task2_recs_')
tasks[3] = os.path.join('task3_qarecs', 'task3_qarecs_pipe_')
tasks[4] = os.path.join('task4_reddit', 'task4_reddit', 'task4_reddit_pipeless_')


def _path(task, opt):
    # Build the data if it doesn't exist.
    build(opt)
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'train'
    elif dt == 'test':
        suffix = 'test'
    elif dt == 'valid':
        suffix = 'dev'

    datafile = os.path.join(
        opt['datapath'],
        'MovieDialog',
        'movie_dialog_dataset',
        '{t}{s}.txt'.format(t=tasks[int(task)], s=suffix),
    )
    if int(task) == 4:
        if dt == 'train':
            candpath = None
        else:
            candpath = datafile.replace(suffix + '.txt', 'cand-{dt}.txt'.format(dt=dt))
    else:
        candpath = os.path.join(
            opt['datapath'], 'MovieDialog', 'movie_dialog_dataset', 'entities.txt'
        )
    return datafile, candpath


# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    """
    Simple text entry with each movie's facts in the knowledge base.
    """

    def __init__(self, opt, shared=None):
        """
        Initialize teacher.
        """
        build(opt)
        opt['datafile'] = os.path.join(
            opt['datapath'], 'MovieDialog', 'movie_dialog_dataset', 'movie_kb.txt'
        )
        super().__init__(opt, shared)


# Single task.
class TaskTeacher(FbDialogTeacher):
    """
    Teacher with single task, specified by moviedialog:task:N.
    """

    def __init__(self, opt, shared=None):
        """
        Initialize teacher.
        """
        try:
            # expecting "moviedialog:task:N"
            self.task = opt['task'].split(':')[2]
        except IndexError:
            self.task = '1'  # default task
        opt['datafile'], opt['cands_datafile'] = _path(self.task, opt)
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(MultiTaskTeacher):
    """
    By default will load teacher with all four tasks.
    """

    def __init__(self, opt, shared=None):
        """
        Initialize teacher.
        """
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(
            'moviedialog:Task:%d' % (i + 1) for i in range(len(tasks))
        )
        super().__init__(opt, shared)

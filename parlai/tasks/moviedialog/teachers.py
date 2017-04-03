#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import copy

from parlai.core.fbdialog import FbDialogTeacher
from parlai.core.agents import MultiTaskTeacher
from .build import build


def _path(task, opt):
    # Build the data if it doesn't exist.
    build(opt)
    tasks = {}
    tasks[1] = 'task1_qa/task1_qa_'
    tasks[2] = 'task2_recs/task2_recs_'
    tasks[3] = 'task3_qarecs/task3_qarecs_'
    tasks[4] = 'task4_reddit/task4_reddit/task4_reddit_'
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'train'
    elif dt == 'test':
        suffix = 'test'
    elif dt == 'valid':
        suffix = 'dev'
    return (opt['datapath'] + 'MovieDialog/movie_dialog_dataset/' +
            tasks[int(task)] + suffix + '.txt')


# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = (opt['datapath'] +
                           'MovieDialog/movie_dialog_dataset/' +
                           'movie_kb.txt')
        super().__init__(opt, shared)


# Single task.
class TaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path(opt['task'].split(':')[2], opt)
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('moviedialog:Task:%d' % (i + 1)
                               for i in range(20))
        super().__init__(opt, shared)

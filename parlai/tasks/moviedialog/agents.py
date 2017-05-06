# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.fbdialog_teacher import FbDialogTeacher
from parlai.core.agents import MultiTaskTeacher
from .build import build

import copy
import os

tasks = {}
tasks[1] = os.path.join('task1_qa', 'task1_qa_')
tasks[2] = os.path.join('task2_recs', 'task2_recs_')
tasks[3] = os.path.join('task3_qarecs', 'task3_qarecs_')
tasks[4] = os.path.join('task4_reddit', 'task4_reddit', 'task4_reddit_')

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

    datafile = os.path.join(opt['datapath'], 'MovieDialog',
                            'movie_dialog_dataset',
                            tasks[int(task)] + suffix + '.txt')
    if int(task) == 4:
        if dt == 'train':
            candpath = None
        else:
            candpath = datafile.replace(suffix + '.txt',
                                        'cand-' + dt + '.txt')
    else:
        candpath = os.path.join(opt['datapath'], 'MovieDialog',
                                'movie_dialog_dataset', 'entities.txt')
    return datafile, candpath


# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'MovieDialog',
                                       'movie_dialog_dataset', 'movie_kb.txt')
        super().__init__(opt, shared)


# Single task.
class TaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'], opt['cands_datafile'] = _path(
            opt['task'].split(':')[2], opt)
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('moviedialog:Task:%d' % (i + 1)
                               for i in range(len(tasks)))
        super().__init__(opt, shared)

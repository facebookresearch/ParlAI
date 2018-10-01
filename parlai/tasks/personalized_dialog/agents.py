#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher
from parlai.core.agents import MultiTaskTeacher
from .build import build

import copy
import os

tasks = {}
tasks[1] = 'personalized-dialog-task1-API-calls'
tasks[2] = 'personalized-dialog-task2-API-refine'
tasks[3] = 'personalized-dialog-task3-options'
tasks[4] = 'personalized-dialog-task4-info'
tasks[5] = 'personalized-dialog-task5-full-dialogs'


def _path(exsz, task, opt):
    # Build the data if it doesn't exist.
    build(opt)
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'trn'
    elif dt == 'test':
        suffix = 'tst'
    elif dt == 'valid':
        suffix = 'dev'
    return os.path.join(
        opt['datapath'], 'personalized-dialog', 'personalized-dialog-dataset',
        '{exsz}'.format(exsz=exsz),
        '{tsk}-{type}.txt'.format(tsk=tasks[int(task)], type=suffix)
    )


# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = os.path.join(
            opt['datapath'], 'personalized-dialog', 'personalized-dialog-dataset',
            'personalized-dialog-kb-all.txt'
        )
        super().__init__(opt, shared)


# python <script.py> -t personalized_dialog:FullTask:<task_id>
# Single full task.
class FullTaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path('full', opt['task'].split(':')[2], opt)
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], 'personalized-dialog', 'personalized-dialog-dataset',
            'personalized-dialog-candidates.txt'
        )
        super().__init__(opt, shared)


# python <script.py> -t personalized_dialog:SmallTask:<task_id>
# Single small task.
class SmallTaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path('small', opt['task'].split(':')[2], opt)
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], 'personalized-dialog', 'personalized-dialog-dataset',
            'personalized-dialog-candidates.txt'
        )
        super().__init__(opt, shared)


# python <script.py> -t personalized_dialog:AllFull
# By default train on all tasks at once.
class AllFullTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('personalized_dialog:FullTask:%d' % (i + 1)
                               for i in range(5))
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], 'personalized-dialog', 'personalized-dialog-dataset',
            'personalized-dialog-candidates.txt'
        )
        super().__init__(opt, shared)


# python <script.py> -t personalized_dialog:AllSmall
# By default train on all tasks at once.
class AllSmallTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('personalized_dialog:SmallTask:%d' % (i + 1)
                               for i in range(5))
        opt['cands_datafile'] = os.path.join(
            opt['datapath'], 'personalized-dialog', 'personalized-dialog-dataset',
            'personalized-dialog-candidates.txt'
        )
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(AllSmallTeacher):
    pass

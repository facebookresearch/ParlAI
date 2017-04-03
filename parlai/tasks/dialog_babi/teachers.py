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
    tasks[1] = 'dialog-babi-task1-API-calls'
    tasks[2] = 'dialog-babi-task2-API-refine'
    tasks[3] = 'dialog-babi-task3-options'
    tasks[4] = 'dialog-babi-task4-phone-address'
    tasks[5] = 'dialog-babi-task5-full-dialogs'
    tasks[6] = 'dialog-babi-task6-dstc2'
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'trn'
    elif dt == 'test':
        suffix = 'tst'
    elif dt == 'valid':
        suffix = 'dev'
    return (opt['datapath'] + 'dialog-bAbI/dialog-bAbI-tasks/' +
            '{tsk}-{type}.txt'.format(
            tsk=tasks[int(task)], type=suffix))


# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = (opt['datapath'] + 'dialog-bAbI/dialog-bAbI-tasks/' +
                           'dialog-babi-kb-all.txt')
        super().__init__(opt, shared)


# Single task.
class TaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path(opt['task'].split(':')[2], opt)
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        # opt['task'] = 'babi:Task:1,babi:Task:2' etc.
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('dialog_babi:Task:%d' % (i + 1)
                               for i in range(6))
        super().__init__(opt, shared)

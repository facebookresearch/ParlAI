#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from parlai.core.teachers import FbDialogTeacher
from parlai.tasks.dialog_babi_plus.build import build

tasks = {}

tasks[1] = 'dialog-babi-plus-task1-API-calls'


def _path(task, opt):
    # Build the data if it doesn't exist.
    build(opt)
    prefix = os.path.join(opt['datapath'], 'dialog-bAbI-plus', 'dialog-bAbI-plus-tasks')
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'trn'
    elif dt == 'test':
        suffix = 'tst'
    elif dt == 'valid':
        suffix = 'dev'
    datafile = os.path.join(
        prefix, '{tsk}-{type}.txt'.format(tsk=tasks[int(task)], type=suffix)
    )

    cands_datafile = os.path.join(prefix, 'dialog-babi-candidates.txt')

    return datafile, cands_datafile


# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = os.path.join(
            opt['datapath'],
            'dialog-bAbI-plus',
            'dialog-bAbI-plus-tasks',
            'dialog-babi-kb-all.txt',
        )
        super().__init__(opt, shared)


# Single task.
class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        default_task_id = 1
        paths = _path(default_task_id, opt)
        opt['datafile'], opt['cands_datafile'] = paths
        super().__init__(opt, shared)

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

def _path(exsz, task, opt, dt=''):
    # Build the data if it doesn't exist.
    build(opt)
    if dt == '':
        dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'bAbI', 'tasks_1-20_v1-2',
                        'en-valid{exsz}-nosf'.format(exsz=exsz),
                        'qa{task}_{type}.txt'.format(task=task, type=dt))


# Single bAbI task (1k training).
class Task1kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = opt.get('task', 'babi:Task1k:1')
        opt['datafile'] = _path('', task.split(':')[2], opt)
        opt['cands_datafile'] = _path('', task.split(':')[2], opt, 'train')
        super().__init__(opt, shared)


# Single bAbI task (10k training).
class Task10kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = opt.get('task', 'babi:Task10k:1')
        opt['datafile'] = _path('-10k', task.split(':')[2], opt)
        opt['cands_datafile'] = _path('-10k', task.split(':')[2], opt, 'train')
        super().__init__(opt, shared)


# By default train on all tasks at once.
class All1kTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('babi:Task1k:%d' % (i + 1) for i in range(20))
        super().__init__(opt, shared)


# By default train on all tasks at once.
class All10kTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('babi:Task10k:%d' % (i + 1) for i in range(20))
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(All1kTeacher):
    pass

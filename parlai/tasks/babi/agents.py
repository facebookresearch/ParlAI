# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.fbdialog_teacher import FbDialogTeacher
from parlai.core.agents import MultiTaskTeacher
from .build import build

import copy
import itertools
import os


def _path(exsz, task, opt, dt=''):
    # Build the data if it doesn't exist.
    build(opt)
    if dt == '':
        dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'bAbI', 'tasks_1-20_v1-2',
                        'en-valid{exsz}-nosf'.format(exsz=exsz),
                        'qa{task}_{type}.txt'.format(task=task, type=dt))


def mod_entry(entry, task):
    y = entry[1]
    if y is not None:
        if task == '8':
            # holding, labels like 'milk,cookies,football'
            # add permutations like 'milk,football,cookies', etc'
            # split = y[0].split(',')
            # entry[1] = [','.join(p) for p in itertools.permutations(split)]
            entry[1] = [y[0].replace(',', '_')]
        elif task == '19':
            # pathfinding, labels like 'n,e' or 's,w'
            # add version with spaces, 'n e'
            # entry[1] = [y[0], y[0].replace(',', ' ')]
            entry[1] = [y[0].replace(',', ' ')]

    return entry


# Single bAbI task (1k training).
class Task1kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = opt.get('task', 'babi:Task1k:1')
        self.task_num = task.split(':')[2]
        opt['datafile'] = _path('', self.task_num, opt)
        opt['cands_datafile'] = _path('', task.split(':')[2], opt, 'train')
        super().__init__(opt, shared)

    def setup_data(self, path):
        for entry, new in super().setup_data(path):
            yield mod_entry(entry, self.task_num), new


# Single bAbI task (10k training).
class Task10kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = opt.get('task', 'babi:Task10k:1')
        self.task_num = task.split(':')[2]
        opt['datafile'] = _path('-10k', self.task_num, opt)
        opt['cands_datafile'] = _path('-10k', task.split(':')[2], opt, 'train')
        super().__init__(opt, shared)

    def setup_data(self, path):
        for entry, new in super().setup_data(path):
            yield mod_entry(entry, self.task_num), new

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

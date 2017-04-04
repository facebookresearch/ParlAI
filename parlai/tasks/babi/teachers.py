# Copyright 2004-present Facebook. All Rights Reserved.

import copy

from parlai.core.fbdialog import FbDialogTeacher
from parlai.core.agents import MultiTaskTeacher
from .build import build


def _path(exsz, task, opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return (opt['datapath'] + '/bAbI/' +
            'tasks_1-20_v1-2/en-valid' +
            '{exsz}-nosf/qa{task}_{type}.txt'.format(
                exsz=exsz, task=task, type=dt))


# Single bAbI task (1k training).
class Task1kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = opt.get('task', 'babi:Task1k:1')
        opt['datafile'] = _path('', task.split(':')[2], opt)
        super().__init__(opt, shared)


# Single bAbI task (10k training).
class Task10kTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = opt.get('task', 'babi:Task-10k:1')
        opt['datafile'] = _path('-10k', task.split(':')[2], opt)
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        # opt['task'] = "babi:Task:1,babi:Task:2" etc.
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('babi:Task1k:%d' % (i + 1) for i in range(20))
        super().__init__(opt, shared)

# Copyright 2004-present Facebook. All Rights Reserved.

from parlai.core.fbdialog import FbDialogTeacher
from .build import build

_tasks = {}
_tasks[1] = 'rl1_pure_imitation'
_tasks[2] = 'rl2_pos_neg'
_tasks[3] = 'rl3_with_ans'
_tasks[4] = 'rl4_with_hint'
_tasks[5] = 'rl5_told_sf'
_tasks[6] = 'rl6_only_some_rewards'
_tasks[7] = 'rl7_no_feedback'
_tasks[8] = 'rl8_imitation_plus_rl'
_tasks[9] = 'rl9_ask_for_answer'
_tasks[10] = 'rl10_ask_for_sf'

_suffixes = {
    'train': 'train',
    'test': 'test',
    'valid': 'dev'
}


def _path(subdir, task, opt):
    build(opt)
    dt = opt['datatype'].split(':')[0]
    task_name = "%s_%s" % (task.split('_')[1],
                           _tasks[int(task.split('_')[0])])
    return (opt['datapath'] + 'DBLL/dbll/' +
            '{subdir}_{task}_{suffix}.txt'.format(
                subdir=subdir, task=task_name,
                suffix=_suffixes[dt]))


# The knowledge base of facts that can be used to answer questions.
class KBTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = (opt['datapath'] + 'DBLL/dbll/movieqa-dbll/' +
                           'movie_kb.txt')
        super().__init__(opt, shared)


# Each individual task.
class TaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        params = opt['task'].split(':')[2]
        opt['datafile'] = _path('movieqa-dbll/movieqa1', params, opt)
        super().__init__(opt, shared)


# Defaults to task 2 with p=0.5.
class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        task = "2_p0.5"
        opt['datafile'] = _path('movieqa-dbll/movieqa1', task, opt)
        super().__init__(opt, shared)

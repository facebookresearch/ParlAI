import os

from parlai.core.fbdialog_teacher import FbDialogTeacher
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
    datafile = os.path.join(prefix,
                            '{tsk}-{type}.txt'.format(tsk=tasks[int(task)], type=suffix))

    cands_datafile = os.path.join(prefix, 'dialog-babi-candidates.txt')

    return datafile, cands_datafile


# Single task.
class TaskTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        paths = _path(opt['task'].split(':')[2], opt)
        opt['datafile'], opt['cands_datafile'] = paths
        super().__init__(opt, shared)

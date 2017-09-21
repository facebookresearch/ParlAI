# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.fbdialog_teacher import FbDialogTeacher
from .build import build

import copy
import os


def _path(task, opt, dt=''):
    # Build the data if it doesn't exist.
    build(opt)
    if dt == '':
        dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'wmt',
                        '{task}_{type}.txt'.format(task=task, type=dt))


class EnDeTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        task = opt.get('task', 'wmt:en_de')
        self.task_name = task.split(':')[1]
        opt['datafile'] = _path(self.task_name, opt, 'train')
        super().__init__(opt, shared)

class DefaultTeacher(EnDeTeacher):
    pass

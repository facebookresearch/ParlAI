#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os


def _path(task, opt, dt):
    # Build the data if it doesn't exist.
    build(opt)
    return os.path.join(opt['datapath'], 'wmt',
                        '{task}_{type}.txt'.format(task=task, type=dt))


class EnDeTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        self.task_name = 'en_de'
        dt = opt['datatype'].split(':')[0]
        opt['datafile'] = _path(self.task_name, opt, dt)
        super().__init__(opt, shared)


class DefaultTeacher(EnDeTeacher):
    pass

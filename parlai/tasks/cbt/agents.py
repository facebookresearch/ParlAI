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


def _path(task, opt):
    # Build the data if it doesn't exist.
    build(opt)
    suffix = ''
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'train'
    elif dt == 'test':
        suffix = 'test_2500ex'
    elif dt == 'valid':
        suffix = 'valid_2000ex'

    return os.path.join(
        opt['datapath'], 'CBT', 'CBTest', 'data', task + '_' + suffix + '.txt')


class NETeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path('cbtest_NE', opt)
        opt['cloze'] = True
        super().__init__(opt, shared)


class CNTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path('cbtest_CN', opt)
        opt['cloze'] = True
        super().__init__(opt, shared)


class VTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path('cbtest_V', opt)
        opt['cloze'] = True
        super().__init__(opt, shared)


class PTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path('cbtest_P', opt)
        opt['cloze'] = True
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = 'cbt:NE,cbt:CN,cbt:V,cbt:P'
        super().__init__(opt, shared)

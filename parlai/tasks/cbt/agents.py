#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDialogTeacher
import parlai.core.agents as core_agents
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
class DefaultTeacher(core_agents.MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = 'cbt:NE,cbt:CN,cbt:V,cbt:P'
        super().__init__(opt, shared)

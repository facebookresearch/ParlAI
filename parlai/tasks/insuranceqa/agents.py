#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import os

from parlai.core.teachers import FbDeprecatedDialogTeacher
from .build import build


def _path(version, opt, exsz=''):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if exsz:
        fname = '%s.%s.txt' % (dt, exsz)
    else:
        fname = '%s.txt' % dt
    return os.path.join(opt['datapath'], 'InsuranceQA', version, fname)


# V1 InsuranceQA task
class V1Teacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path('V1', opt)
        super().__init__(opt, shared)


# V2 InsuranceQA task
class V2Teacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        task = opt.get('task', None)
        if not task:
            # options are 100, 500, 1000, or 1500
            task = 'insuranceqa:V2:100'
        split = task.split(':')
        opt['datafile'] = _path('V2', opt, split[2])
        super().__init__(opt, shared)


class DefaultTeacher(V1Teacher):
    pass

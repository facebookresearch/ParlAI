# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import copy

from parlai.core.fbdialog import FbDialogTeacher
from .build import build


def _path(opt, filtered):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return '{}/MCTest/{}{}.txt'.format(opt['datapath'], dt, filtered)


class Task160Teacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '160')
        super().__init__(opt, shared)


class Task500Teacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '500')
        super().__init__(opt, shared)


class DefaultTeacher(Task500Teacher):
    pass

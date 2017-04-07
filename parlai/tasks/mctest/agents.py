#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

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

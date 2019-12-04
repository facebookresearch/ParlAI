#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from .build import build, make_path
from parlai.utils.misc import warn_once
from parlai.core.teachers import ParlAIDialogTeacher


def _path(opt):
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    if datatype == 'test':
        warn_once("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    return make_path(opt, datatype + '.txt')


class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt)
        super().__init__(opt, shared)

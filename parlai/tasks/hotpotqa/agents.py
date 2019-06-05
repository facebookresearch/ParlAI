#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import ParlAIDialogTeacher
from parlai.core.utils import warn_once
from .build import build

import copy
import os


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt == 'test_distractor':
        warn_once('WARNING: Test set not included. Setting datatype to valid.')
        dt = 'valid_distractor'
    if dt == 'test_fullwiki':
        warn_once('WARNING: Test set not included. Setting datatype to valid.')
        dt = 'valid_fullwiki'
    return os.path.join(opt['datapath'], 'HotpotQA', dt + '.txt')


class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt)
        super().__init__(opt, shared)

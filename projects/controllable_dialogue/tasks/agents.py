#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from .build import build, make_path
from parlai.core.teachers import ParlAIDialogTeacher


def _path(opt, filtered):
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return make_path(opt, dt + '.txt')


class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        # get datafile
        opt['datafile'] = _path(opt, '')

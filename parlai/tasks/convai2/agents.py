# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os

def _path(opt, persona):
    # Build the data if it doesn't exist.
    build(opt)
    datatype =  opt['datatype'].split(':')[0]
    if datatype == 'test':
        print("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    dt = datatype + '_' + persona
    return os.path.join(opt['datapath'], 'ConvAI2', dt + '.txt')

class NoneTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'none_original')
        super().__init__(opt, shared)

class SelfOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_original')
        super().__init__(opt, shared)

class SelfTeacher(SelfOriginalTeacher):
    pass

class SelfRevisedTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_revised')
        super().__init__(opt, shared)

class DefaultTeacher(SelfOriginalTeacher):
    pass

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDeprecatedDialogTeacher
from .build import build

import copy
import os


def _path(opt):
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt == 'train':
        suffix = 'train.14M+.txt'
    elif dt == 'valid':
        suffix = 'validation_NECN.20k.txt'
    else:
        suffix = 'test_CN.10k.txt'
    return os.path.join(opt['datapath'], 'BookTest', 'booktest-gut', suffix)


class DefaultTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt)
        if 'stream' not in opt['datatype']:
            print(
                'Dataset might not fit in memory. If this is the case, use'
                + ' streaming by setting "-dt '
                + opt['datatype']
                + ':stream".'
            )
        super().__init__(opt, shared)

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import DialogTeacher
from parlai.tasks.multinli.agents import setup_data
from .build import build

import os
import copy
import json


SNLI = 'SNLI'
SNLI_VERSION = '1.0'
SNLI_PREFIX = 'snli_'


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        suffix = 'train'
    elif dt == 'valid':
        suffix = 'dev'
    elif dt == 'test':
        suffix = 'test'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], SNLI,
                             SNLI_PREFIX + SNLI_VERSION,
                             SNLI_PREFIX + SNLI_VERSION +
                             '_' + suffix + '.jsonl')

    return data_path


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'SNLI'
        super().__init__(opt, shared)

    def setup_data(self, path):
        return setup_data(path)

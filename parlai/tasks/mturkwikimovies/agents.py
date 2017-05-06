# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.fbdialog_teacher import FbDialogTeacher
from .build import build

import copy
import os

def _path(opt, filtered):
    # Build the data if it doesn't exist. It depends on wikimovies.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt == 'valid':
        dt = 'dev'
    return os.path.join(opt['datapath'], 'MTurkWikiMovies', 'mturkwikimovies',
                        'qa-{type}.txt'.format(type=dt))


class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        opt['cands_datafile'] = os.path.join(opt['datapath'], 'WikiMovies',
                                             'movieqa', 'knowledge_source',
                                             'entities.txt')
        super().__init__(opt, shared)

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import ParlAIDialogTeacher
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


class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        opt['cands_datafile'] = os.path.join(opt['datapath'], 'WikiMovies',
                                             'movieqa', 'knowledge_source',
                                             'entities.txt')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        opt['parlaidialogteacher_cands_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['cands_datafile']
        )
        super().__init__(opt, shared)

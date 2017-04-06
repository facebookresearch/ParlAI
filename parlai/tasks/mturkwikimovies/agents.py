#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import copy

from parlai.core.fbdialog import FbDialogTeacher
import parlai.tasks.wikimovies.build as wikimovies_build
from .build import build


def _path(opt, filtered):
    # Build the data if it doesn't exist. It depends on wikimovies.
    wikimovies_build.build(opt)
    build(opt)
    dt = opt['datatype']
    if dt == 'valid':
        dt = 'dev'
    return (opt['datapath'] + '/MTurkWikiMovies/mturkwikimovies/' +
            'qa-{type}.txt'.format(type=dt))


class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        super().__init__(opt, shared)

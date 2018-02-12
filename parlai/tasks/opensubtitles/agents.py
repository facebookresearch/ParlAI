# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher
from .build_2009 import build as build_2009
from .build_2018 import build as build_2018

import copy
import os


def _path(opt, version):
    # Build the data if it doesn't exist.
    if version == '2009':
        datapath = build_2009(opt['datapath'])
    elif version == '2018':
        datapath = build_2018(opt['datapath'])
    else:
        raise Exception('Unknown version for OpenSubtitles: %s' % version)
    return os.path.join(datapath, opt['datatype'].split(':')[0] + '.txt')


class DefaultTeacher(FbDialogTeacher):
    """This version of opensubtitles creates half of all possible dialog
    examples.
    """
    def __init__(self, opt, shared=None, version='2018'):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, version)
        if not opt['datatype'].startswith('train'):
            opt['cands_datafile'] = opt['datafile']
        super().__init__(opt, shared)


class V2009Teacher(DefaultTeacher):
    """This version of opensubtitles creates all possible dialog examples."""

    def __init__(self, opt, shared=None):
        super(V2009Teacher, self).__init__(opt, shared, '2009')

    def setup_data(self, path):
        alternate = []
        for entry, new in super().setup_data(path):
            if new:
                for i, e in enumerate(alternate):
                    yield e, i == 0
                alternate.clear()
            else:
                alternate.append(entry)
            yield entry, new
        if alternate:
            for i, e in enumerate(alternate):
                yield e, i == 0


class Task100kTeacher(DefaultTeacher):
    """This version of opensubtitles only includes 100,000 dialogs."""
    def setup_data(self, path):
        cnt = 0
        for entry, new in super().setup_data(path):
            if cnt < 100000:
                yield entry, new
            cnt += 1


class Task10kTeacher(DefaultTeacher):
    """This version of opensubtitles only includes 10,000 dialogs."""
    def setup_data(self, path):
        cnt = 0
        for entry, new in super().setup_data(path):
            if cnt < 10000:
                yield entry, new
            cnt += 1

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os


def _path(opt, filtered):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'OpenSubtitles2016',
                        dt + filtered + '.txt')


class HalfTeacher(FbDialogTeacher):
    """This version of opensubtitles creates half of all possible dialog
    examples.
    """
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        if not opt['datatype'].startswith('train'):
            opt['cands_datafile'] = opt['datafile']
        super().__init__(opt, shared)


class FullTeacher(HalfTeacher):
    """This version of opensubtitles creates all possible dialog examples."""
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


class Task100kTeacher(HalfTeacher):
    """This version of opensubtitles only includes 100,000 dialogs."""
    def setup_data(self, path):
        cnt = 0
        for entry, new in super().setup_data(path):
            if cnt < 100000:
                yield entry, new
            cnt += 1


class Task10kTeacher(HalfTeacher):
    """This version of opensubtitles only includes 10,000 dialogs."""
    def setup_data(self, path):
        cnt = 0
        for entry, new in super().setup_data(path):
            if cnt < 10000:
                yield entry, new
            cnt += 1


# Defaults to full teacher (all possible examples)
class DefaultTeacher(FullTeacher):
    pass

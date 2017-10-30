# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.fbdialog_teacher import FbDialogTeacher
from .build import build

import copy
import os
import re


def _regularize(sent):
    sent = sent.replace('i&gt;', '').replace('&lt;', '').replace('&gt;', '')
    sent = re.sub(r'x[0-9|a-z][0-9]', ' ', sent)
    sent = sent.replace('\\', '')
    sent = ' '.join(re.findall(r"[\w']+|[.,!?:;]", sent))
    return sent


def _path(opt, filtered):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'OpenSubtitles',
                        dt + filtered + '.txt')


class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        opt['cands_datafile'] = opt['datafile']
        super().__init__(opt, shared)

    def setup_data(self, path):
        alternate = []
        for entry, new in super().setup_data(path):
            entry[0] = _regularize(entry[0])
            if len(entry) > 1 and entry[1]:  # labels
                entry[1] = [_regularize(y) for y in entry[1]]
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

    def load_cands(self, path):
        return [r for r in (_regularize(c) for c in super().load_cands(path)) if r]

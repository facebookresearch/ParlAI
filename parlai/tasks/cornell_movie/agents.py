#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os


def _path(opt, filtered):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'CornellMovie',
                        dt + filtered + '.txt')


class DefaultTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        opt['cands_datafile'] = opt['datafile']
        super().__init__(opt, shared)


class DoubleTeacher(DefaultTeacher):
    """This version creates text-label pairs from the perspective of both
    speakers.
    """
    def setup_data(self, path):
        """Adds additional perspectives.
        For example, in the conversation:

        x1 y1
        x2 y2
        x3

        Creates the additional dialog:

        '' x1
        y1 x2
        y2 x3

        And if a y3 was available in response to x3, also would have added:

        y3
        """

        def rebuild(entries):
            new_list = []
            if len(entries) > 0:
                # prepend silent input => x_0
                new_list.append(('', [entries[0][0]]))

                # add all ( y_t => x_(t+1) ) pairs
                new_list.extend([(entries[i][1][0], [entries[i + 1][0]])
                                 for i in range(len(entries) - 1)])
                if len(entries[-1]) > 1 and entries[-1][1]:
                    # add y_n => '', if last y avail
                    new_list.append((entries[-1][1][0], None))
            return new_list

        # this shows conversations in both directions
        alternate = []
        for entry, new in super().setup_data(path):
            if new:
                for i, e in enumerate(rebuild(alternate)):
                    yield e, i == 0
                alternate.clear()
            alternate.append(entry)
            yield entry, new
        if alternate:
            for i, e in enumerate(rebuild(alternate)):
                yield e, i == 0

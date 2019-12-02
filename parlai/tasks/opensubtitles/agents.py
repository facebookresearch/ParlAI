#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDialogTeacher
from .build_2009 import build as build_2009
from .build_2018 import build as build_2018

import copy
import os

SILENCE_TOKEN = '__SILENCE__'


def _path(opt, version, use_history):
    # Build the data if it doesn't exist.
    if version == '2009':
        assert use_history
        datapath = build_2009(opt['datapath'])
    elif version == '2018':
        datapath = build_2018(opt['datapath'], use_history)
    else:

        raise Exception('Unknown version for OpenSubtitles: %s' % version)
    return os.path.join(datapath, opt['datatype'].split(':')[0] + '.txt')


class HalfTeacher(FbDialogTeacher):
    """
    This version of opensubtitles creates half of all possible dialog examples.
    """

    def __init__(self, opt, shared=None, version='2018', use_history=True):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, version, use_history)
        if not opt['datatype'].startswith('train'):
            opt['cands_datafile'] = opt['datafile']
        super().__init__(opt, shared)

    def setup_data(self, path):
        for entry, new in super().setup_data(path):
            # check that the label is present, else skip this example
            if entry[1]:
                yield entry, new


class FullTeacher(FbDialogTeacher):
    """
    This version of opensubtitles creates all possible dialog examples.
    """

    def __init__(self, opt, shared=None, version='2018', use_history=True):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, version, use_history)
        if not opt['datatype'].startswith('train'):
            opt['cands_datafile'] = opt['datafile']
        super().__init__(opt, shared)

    def setup_data(self, path):
        def rebuild(entries):
            if len(entries) == 0:
                return []
            # flip the first example
            flipped = [(SILENCE_TOKEN, [entries[0][0]], 0)]
            # flip the rest
            flipped += [
                (entries[i][1][0], [entries[i + 1][0]], 0)
                for i in range(len(entries) - 1)
            ]
            return flipped

        # this shows conversations in both directions
        # we skip examples for which no label is present
        alternate = []
        for entry, new in super().setup_data(path):
            if new:
                for i, e in enumerate(rebuild(alternate)):
                    if e[1]:
                        yield e, i == 0
                alternate.clear()
            else:
                alternate.append(entry)
            if entry[1]:
                yield entry, new

        # flip the last episode
        if alternate:
            for i, e in enumerate(rebuild(alternate)):
                if e[1]:
                    yield e, i == 0
            alternate.clear()


class Task100kTeacher(HalfTeacher):
    """
    This version of opensubtitles only includes 100,000 dialogs.
    """

    def setup_data(self, path):
        cnt = 0
        for entry, new in super().setup_data(path):
            if len(entry) > 1 and entry[1]:
                # focus on examples with targets for small set
                yield entry, new
            cnt += 1
            if cnt >= 100000:
                break


class Task10kTeacher(HalfTeacher):
    """
    This version of opensubtitles only includes 10,000 dialogs.
    """

    def setup_data(self, path):
        cnt = 0
        for entry, new in super().setup_data(path):
            if len(entry) > 1 and entry[1]:
                # focus on examples with targets for small set
                yield entry, new
            cnt += 1
            if cnt >= 10000:
                break


class V2009Teacher(FullTeacher):
    def __init__(self, opt, shared=None):
        super(V2009Teacher, self).__init__(opt, shared, '2009', True)


class V2009HalfTeacher(HalfTeacher):
    def __init__(self, opt, shared=None):
        super(V2009HalfTeacher, self).__init__(opt, shared, '2009', True)


class V2009Task100kTeacher(Task100kTeacher):
    def __init__(self, opt, shared=None):
        super(V2009Task100kTeacher, self).__init__(opt, shared, '2009', True)


class V2009Task10kTeacher(Task10kTeacher):
    def __init__(self, opt, shared=None):
        super(V2009Task10kTeacher, self).__init__(opt, shared, '2009', True)


class V2018Teacher(FullTeacher):
    def __init__(self, opt, shared=None):
        super(V2018Teacher, self).__init__(opt, shared, '2018', True)


class V2018HalfTeacher(HalfTeacher):
    def __init__(self, opt, shared=None):
        super(V2018HalfTeacher, self).__init__(opt, shared, '2018', True)


class V2018Task100kTeacher(Task100kTeacher):
    def __init__(self, opt, shared=None):
        super(V2018Task100kTeacher, self).__init__(opt, shared, '2018', True)


class V2018Task10kTeacher(Task10kTeacher):
    def __init__(self, opt, shared=None):
        super(V2018Task10kTeacher, self).__init__(opt, shared, '2018', True)


class V2018NoHistoryTeacher(FullTeacher):
    def __init__(self, opt, shared=None):
        super(V2018NoHistoryTeacher, self).__init__(opt, shared, '2018', False)


class V2018NoHistoryTask100kTeacher(Task100kTeacher):
    """
    Note, these versions only uses two-turns dialog.

    This is more efficient due to movie-based deduplication, compared to the regular
    v2018 dataset.
    """

    def __init__(self, opt, shared=None):
        super(V2018NoHistoryTask100kTeacher, self).__init__(opt, shared, '2018', False)


class V2018NoHistoryTask10kTeacher(Task10kTeacher):
    def __init__(self, opt, shared=None):
        super(V2018NoHistoryTask10kTeacher, self).__init__(opt, shared, '2018', False)


# Defaults to full teacher (all possible examples)
class DefaultTeacher(V2018Teacher):
    pass

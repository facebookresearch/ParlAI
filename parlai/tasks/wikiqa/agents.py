# Copyright 2004-present Facebook. All Rights Reserved.

import copy

from parlai.core.fbdialog import FbDialogTeacher
from .build import build


def _path(opt, filtered):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return (opt['datapath'] + '/WikiQA/' +
            '{type}.txt'.format(type=dt + filtered))


class FilteredTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '-filtered')
        super().__init__(opt, shared)


class UnfilteredTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        super().__init__(opt, shared)


class DefaultTeacher(FilteredTeacher):
    pass

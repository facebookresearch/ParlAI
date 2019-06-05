#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""PersonaChat task agents.

Persona can be 'none', 'self', 'other', or 'both'.
Format of persona can be 'original' or 'revised'.

This is specified in the following way:
--task personachat:{format}
...where {format} is one of...
- none
- self_original
- self_revised
- other_original
- other_revised
- both_original
- both_revised
"""


from parlai.core.teachers import ParlAIDialogTeacher
from .build import build

import copy
import os


def _path(opt, persona):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0] + '_' + persona
    return os.path.join(opt['datapath'], 'Persona-Chat', 'personachat', dt + '.txt')


class NoneTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'none_original')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        super().__init__(opt, shared)


class SelfOriginalTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_original')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        super().__init__(opt, shared)


class SelfTeacher(SelfOriginalTeacher):
    pass


class SelfRevisedTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_revised')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        super().__init__(opt, shared)


class OtherOriginalTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'other_original')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        super().__init__(opt, shared)


class OtherTeacher(OtherOriginalTeacher):
    pass


class OtherRevisedTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'other_revised')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        super().__init__(opt, shared)


class BothOriginalTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'both_original')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        super().__init__(opt, shared)


class BothTeacher(BothOriginalTeacher):
    pass


class BothRevisedTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'both_revised')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        super().__init__(opt, shared)


class DefaultTeacher(SelfOriginalTeacher):
    pass

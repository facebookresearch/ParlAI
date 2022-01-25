#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from .build import build
from parlai.core.teachers import FbDeprecatedDialogTeacher

'''This dataset is available in seven different languages.
To use the dataset in the specified language, use the task flag to specify it.

--task xpersona:{LANGUAGE}

The default language is English.
'''


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    if ':' in opt['task']:
        language = opt['task'].split(':')[1]
    else:
        language = 'En'
    return os.path.join(opt['datapath'], 'XPersona', language + '_' + dt + '.txt')


class XPersonaTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt)
        super().__init__(opt, shared)


class DefaultTeacher(XPersonaTeacher):
    pass


class EnTeacher(XPersonaTeacher):
    pass


class ItTeacher(XPersonaTeacher):
    pass


class ZhTeacher(XPersonaTeacher):
    pass


class IdTeacher(XPersonaTeacher):
    pass


class FrTeacher(XPersonaTeacher):
    pass


class JpTeacher(XPersonaTeacher):
    pass


class KoTeacher(XPersonaTeacher):
    pass

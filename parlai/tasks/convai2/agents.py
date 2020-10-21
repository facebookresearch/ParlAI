#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDeprecatedDialogTeacher
from parlai.utils.misc import warn_once
from .build import build
from parlai.utils.strings import normalize_reply
import parlai.utils.logging as logging

import copy
import os

'''All teachers have a version with and without label candidates. Each teacher
defaults to using a dataset with label candidates. To use a dataset without
label candidates, specify this using the task flag:

--task convai2:{TEACHER_NAME}:no_cands

where TEACHER_NAME is None, SelfOriginal (Self), or SelfRevised.
'''


def _path(opt, persona, use_cands):
    # Build the data if it doesn't exist.
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    if datatype == 'test':
        warn_once("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    dt = datatype + '_' + persona
    cands = '' if use_cands else '_no_cands'
    return os.path.join(opt['datapath'], 'ConvAI2', dt + cands + '.txt')


class BothTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_original', use_cands)
        super().__init__(opt, shared)


class NoneTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'none_original', use_cands)
        super().__init__(opt, shared)


class SelfOriginalTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_original', use_cands)
        super().__init__(opt, shared)


class SelfTeacher(SelfOriginalTeacher):
    pass


class SelfRevisedTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_revised', use_cands)
        super().__init__(opt, shared)


class NormalizedTeacher(SelfOriginalTeacher):
    def normalize_replies(self, x):
        xs = x.split('\n')
        xs2 = []
        for x in xs:
            if 'your persona:' in x:
                # Normalize the sentence appearing after 'your persona:'
                x = x[len('your persona: ') :]
                x = normalize_reply(x)
                x = 'your persona: ' + x
            else:
                x = normalize_reply(x)

            xs2.append(x)
        return '\n'.join(xs2)

    def setup_data(self, path):
        logging.info(f"loading normalized fbdialog data: {path}")
        for (text, labels, reward, candidates), new_episode in super().setup_data(path):
            text = self.normalize_replies(text)
            labels = [self.normalize_replies(l) for l in labels]
            candidates = [self.normalize_replies(c) for c in candidates]
            yield (text, labels, reward, candidates), new_episode


class DefaultTeacher(SelfOriginalTeacher):
    pass


class InteractiveTeacher(SelfOriginalTeacher):
    # Dummy class to add arguments for interactive world.
    pass


class SelfchatTeacher(SelfOriginalTeacher):
    # Dummy class to add arguments for interactive world.
    pass

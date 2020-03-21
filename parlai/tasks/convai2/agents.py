#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import FbDialogTeacher
from parlai.utils.misc import warn_once
from .build import build
from parlai.utils.strings import normalize_reply

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


class BothTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_original', use_cands)
        super().__init__(opt, shared)


class NoneTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except Exception:
            use_cands = True
        opt['datafile'] = _path(opt, 'none_original', use_cands)
        super().__init__(opt, shared)


class SelfOriginalTeacher(FbDialogTeacher):
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


class SelfRevisedTeacher(FbDialogTeacher):
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
            xs2.append(normalize_reply(x))
        return '\n'.join(xs2)

    def setup_data(self, path):
        print("[loading normalized fbdialog data:" + path + "]")
        with open(path) as read:
            start = True
            x = ''
            reward = 0
            last_conv_id = None
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) == 0:
                    # empty response
                    continue

                # first, get conversation index -- '1' means start of episode
                space_idx = line.find(' ')
                if space_idx == -1:
                    # empty line, both individuals are saying whitespace
                    conv_id = int(line)
                else:
                    conv_id = int(line[:space_idx])

                # split line into constituent parts, if available:
                # x<tab>y<tab>reward<tab>label_candidates
                # where y, reward, and label_candidates are optional
                split = line[space_idx + 1 :].split('\t')

                # remove empty items and strip each one
                for i in range(len(split)):
                    word = split[i].strip()
                    if len(word) == 0:
                        split[i] = ''
                    else:
                        split[i] = word
                # Empty reward string same as None
                if len(split) > 2 and split[2] == '':
                    split[2] = None

                # now check if we're at a new episode
                if last_conv_id is None or conv_id <= last_conv_id:
                    x = x.strip()
                    if x:
                        x = self.normalize_replies(x)
                        import pdb

                        pdb.set_trace()
                        yield [x, None, reward], start
                    start = True
                    reward = 0
                    # start a new episode
                    x = split[0]
                else:
                    if x:
                        # otherwise add current x to what we have so far
                        x = '{x}\n{next_x}'.format(x=x, next_x=split[0])
                    else:
                        x = split[0]
                last_conv_id = conv_id
                if len(split) > 2 and split[2]:
                    reward += float(split[2])

                if len(split) > 1 and split[1]:
                    # only generate an example if we have a y
                    split[0] = x
                    # split labels
                    split[1] = split[1].split('|')
                    if len(split) > 3:
                        # split label_candidates
                        split[3] = split[3].split('|')
                    if len(split) > 2:
                        split[2] = reward
                    else:
                        split.append(reward)
                    # normalize
                    split[0] = self.normalize_replies(split[0])
                    for i, _c in enumerate(split[1]):
                        split[1][i] = self.normalize_replies(split[1][i])
                    for i, _c in enumerate(split[3]):
                        split[3][i] = self.normalize_replies(split[3][i])
                    if start:
                        yield split, True
                        start = False
                    else:
                        yield split, False
                    # reset x in case there is unlabeled data still left
                    x = ''
                    reward = 0
            if x:
                x = self.normalize_replies(x)
                yield [x, None, reward], start


class DefaultTeacher(SelfOriginalTeacher):
    pass


class InteractiveTeacher(SelfOriginalTeacher):
    # Dummy class to add arguments for interactive world.
    pass


class SelfchatTeacher(SelfOriginalTeacher):
    # Dummy class to add arguments for interactive world.
    pass

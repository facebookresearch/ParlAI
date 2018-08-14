# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.teachers import FbDialogTeacher, ParlAIDialogTeacher
from .build import build

import copy
import os

'''This Teacher is based on the dataset in the ParlAI Dialog format,
which was collected during the wild evaluation of ConvaAI2 participants bots.'''


def _path(opt, filtered):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'ConvAI2_wild_evaluation', dt + '.txt')


class DefaultTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)

        # get datafile
        opt['datafile'] = _path(opt, '')



"""
def _path(opt, persona, use_cands):
    # Build the data if it doesn't exist.
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    if datatype == 'test':
        print("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    dt = datatype + '_' + persona
    cands = '' if use_cands else '_no_cands'
    return os.path.join(opt['datapath'], 'ConvAI2_wild_eval', dt + cands + '.txt')


class NoneTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'none_original', use_cands)
        super().__init__(opt, shared)


class SelfOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_original', use_cands)
        super().__init__(opt, shared)


class SelfTeacher(SelfOriginalTeacher):
    pass
"""

#class SelfRevisedTeacher(FbDialogTeacher):
#    def __init__(self, opt, shared=None):
#        opt = copy.deepcopy(opt)
#        try:
#            cands = opt['task'].split(":")[2]
#            use_cands = False if cands == 'no_cands' else True
#        except:
#            use_cands = True
#        opt['datafile'] = _path(opt, 'self_revised', use_cands)
#        super().__init__(opt, shared)

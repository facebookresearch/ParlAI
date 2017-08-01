# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

from parlai.core.fbdialog_teacher import FbDialogTeacher

import copy
import os

class DefaultTeacher(FbDialogTeacher):
    """This task simply loads the specified file: useful for quick tests without
    setting up a new task.
    """
    
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile_datapath', type=str,
                           help="Data file in FbDialogFormat")

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath'):
            raise RuntimeError('fromfile_datapath not specified')
        opt['datafile'] = opt['fromfile_datapath']
        super().__init__(opt, shared)

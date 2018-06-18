# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

from parlai.core.teachers import FbDialogTeacher, ParlAIDialogTeacher
from parlai.core.utils import str_to_msg

import copy

class FbformatTeacher(FbDialogTeacher):
    """This task simply loads the specified file: useful for quick tests without
    setting up a new task.
    """
    
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath', type=str,
                           help="Data file")

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath'):
            raise RuntimeError('fromfile_datapath not specified')
        opt['datafile'] = opt['fromfile_datapath']
        super().__init__(opt, shared)


class Fbformat2Teacher(FbDialogTeacher):
    """This task simply loads the specified file: useful for quick tests without
    setting up a new task. Used to set up a second task.
    """
    
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath2', type=str,
                           help="Data file")
        
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath2'):
            raise RuntimeError('fromfile_datapath2 not specified')
        opt['datafile'] = opt['fromfile_datapath2']
        super().__init__(opt, shared)
        

        
class ParlaiformatTeacher(ParlAIDialogTeacher):
    """This module provides access to data in the ParlAI Text Dialog format.
    See core/teachers.py for more info about the format.
    """
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath', type=str,
                           help="Data file")

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath'):
            raise RuntimeError('fromfile_datapath not specified')
        datafile = opt['fromfile_datapath']
        if shared is None:
            self._setup_data(datafile)
        self.id = datafile
        self.reset()

class Parlaiformat2Teacher(ParlAIDialogTeacher):
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath2', type=str,
                           help="Data file")

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath2'):
            raise RuntimeError('fromfile_datapath2 not specified')
        datafile = opt['fromfile_datapath2']
        if shared is None:
            self._setup_data(datafile)
        self.id = datafile
        self.reset()

class DefaultTeacher(FbformatTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

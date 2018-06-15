# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
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
        

        
class ParlaiformatTeacher(FixedDialogTeacher):
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
        
    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        return self.examples[episode_idx]

    def _setup_data(self, path):
        print("[loading parlAI text data:" + path + "]")
        self.examples = []
        with open(path) as read:
            for line in read:
                msg = str_to_msg(line.rstrip('\n'))
                if msg:
                    self.examples.append(msg)

class Parlaiformat2Teacher(FixedDialogTeacher):
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
        
    def num_examples(self):
        return len(self.examples)

    def num_episodes(self):
        return self.num_examples()

    def get(self, episode_idx, entry_idx=None):
        return self.examples[episode_idx]

    def _setup_data(self, path):
        print("[loading parlAI text data:" + path + "]")
        self.examples = []
        with open(path) as read:
            for line in read:
                msg = str_to_msg(line.rstrip('\n'))
                if msg:
                    self.examples.append(msg)
                    
class DefaultTeacher(FbformatTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

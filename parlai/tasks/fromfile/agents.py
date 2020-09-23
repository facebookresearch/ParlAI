#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

import copy
import os

import parlai.utils.logging as logging
from parlai.core.teachers import FbDeprecatedDialogTeacher, ParlAIDialogTeacher


class FbformatTeacher(FbDeprecatedDialogTeacher):
    """
    This task simply loads the specified file: useful for quick tests without setting up
    a new task.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('-dp', '--fromfile-datapath', type=str, help="Data file")

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath'):
            raise RuntimeError('fromfile_datapath not specified')
        opt['datafile'] = opt['fromfile_datapath']
        super().__init__(opt, shared)


class Fbformat2Teacher(FbDeprecatedDialogTeacher):
    """
    This task simply loads the specified file: useful for quick tests without setting up
    a new task.

    Used to set up a second task.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('-dp', '--fromfile-datapath2', type=str, help="Data file")

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath2'):
            raise RuntimeError('-dp', 'fromfile_datapath2 not specified')
        opt['datafile'] = opt['fromfile_datapath2']
        super().__init__(opt, shared)


class ParlaiformatTeacher(ParlAIDialogTeacher):
    """
    This module provides access to data in the ParlAI Text Dialog format.

    See core/teachers.py for more info about the format.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('-ffdp', '--fromfile-datapath', type=str, help="Data file")
        agent.add_argument(
            '-ffdt',
            '--fromfile-datatype-extension',
            type='bool',
            default=False,
            help="If true, use _train.txt, _valid.txt, _test.txt file extensions",
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        if not opt.get('fromfile_datapath'):
            raise RuntimeError('fromfile_datapath not specified')
        datafile = opt['fromfile_datapath']
        if self.opt['fromfile_datatype_extension']:
            datafile += "_" + self.opt['datatype'].split(':')[0] + '.txt'
        else:
            if shared is None and (
                'valid' in self.opt['datatype'] or 'test' in self.opt['datatype']
            ):
                logging.warn(
                    'You are using this fromfile data as a valid or test set without setting fromfile_datatype_extension to true. Please be aware this uses directly the file you indicated, make sure this is not the same as your training file.'
                )
        if shared is None:
            self._setup_data(datafile)
        # Truncate datafile to just the immediate enclosing folder name and file name
        dirname, basename = os.path.split(datafile)
        self.id = os.path.join(os.path.split(dirname)[1], basename)
        self.reset()


class Parlaiformat2Teacher(ParlAIDialogTeacher):
    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('FromFile Task Arguments')
        agent.add_argument('--fromfile-datapath2', type=str, help="Data file")

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


class DefaultTeacher(ParlaiformatTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

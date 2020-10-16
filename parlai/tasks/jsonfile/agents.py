#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

import copy
import os

from parlai.core.teachers import ConversationTeacher


class JsonTeacher(ConversationTeacher):
    """
    This module provides access to data in the Conversations format.

    See core/teachers.py for more info about the format.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('JsonFile Task Arguments')
        agent.add_argument('-jfdp', '--jsonfile-datapath', type=str, help="Data file")
        agent.add_argument(
            '-jfdt',
            '--jsonfile-datatype-extension',
            type='bool',
            default=False,
            help="If true, use _train.jsonl, _valid.jsonl, _test.jsonl file extensions",
        )
        agent.add_argument(
            '--label-turns',
            type=str,
            help='which speaker to use as label',
            choices=['firstspeaker', 'secondspeaker', 'both'],
            default='secondspeaker',
        )

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        if not opt.get('jsonfile_datapath'):
            raise RuntimeError('jsonfile_datapath not specified')
        datafile = opt['jsonfile_datapath']
        if self.opt['jsonfile_datatype_extension']:
            datafile += "_" + self.opt['datatype'].split(':')[0] + '.jsonl'
        if shared is None:
            self._setup_data(datafile)
        # Truncate datafile to just the immediate enclosing folder name and file name
        dirname, basename = os.path.split(datafile)
        self.id = os.path.join(os.path.split(dirname)[1], basename)
        self.reset()


class DefaultTeacher(JsonTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

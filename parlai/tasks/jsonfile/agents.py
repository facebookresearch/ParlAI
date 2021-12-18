#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import copy
import os

from parlai.core.teachers import ConversationTeacher


class JsonTeacher(ConversationTeacher):
    """
    This module provides access to data in the Conversations format.

    See core/teachers.py for more info about the format.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('JsonFile Task Arguments')
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
        return parser

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        if not opt.get('jsonfile_datapath'):
            raise RuntimeError('jsonfile_datapath not specified')
        datafile = opt['jsonfile_datapath']
        if opt['jsonfile_datatype_extension']:
            datafile += "_" + opt['datatype'].split(':')[0] + '.jsonl'
        opt['conversationteacher_datafile'] = datafile
        super().__init__(opt, shared)

        # Truncate datafile to just the immediate enclosing folder name and file name
        dirname, basename = os.path.split(datafile)
        self.id = os.path.join(os.path.split(dirname)[1], basename)


class DefaultTeacher(JsonTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

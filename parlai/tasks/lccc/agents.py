#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import copy
import os
from .build import build

from parlai.core.teachers import ConversationTeacher


def _path(opt):
    # build the data if it does not exist
    build(opt)

    # set up path to data (specific to each dataset)
    datatype = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'LCCC', datatype + '.json')


class LCCCTeacher(ConversationTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('LCCC Task Arguments')
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
        # get datafile
        opt['conversationteacher_datafile'] = _path(opt)
        super().__init__(opt, shared)


class DefaultTeacher(LCCCTeacher):
    pass

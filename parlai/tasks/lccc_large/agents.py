#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This task simply loads the specified file: useful for quick tests without
# setting up a new task.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
import copy
import os
from .build import build

from parlai.core.teachers import ConversationTeacher


def _path(opt):
    # build the data if it does not exist
    build(opt)
    # set up path to data
    datatype = opt['datatype'].split(':')[0]
    if datatype != 'train':
        warn_once("WARNING: Test set or valid set not included. Setting datatype to train.")
        datatype = 'train'
    return os.path.join(opt['datapath'], 'LCCC_large', 'LCCC_large' + '.json')


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

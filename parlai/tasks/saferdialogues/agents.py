#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import os
import copy
from parlai.core.teachers import ParlAIDialogTeacher
from .build import build


def _path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(
        opt['datapath'], 'saferdialogues', 'saferdialogues_dataset', dt + '.txt'
    )


def _bad_path(opt):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    if dt == 'valid' or dt == 'test':
        dt += '_bad'
    return os.path.join(
        opt['datapath'], 'saferdialogues', 'saferdialogues_dataset', dt + '.txt'
    )


class SaferDialoguesTeacher(ParlAIDialogTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('SaFeRDialogues options')
        agent.add_argument(
            '--recovery',
            type=bool,
            default=True,
            help="Whether or not to include the recovery utterance",
        )
        return parser

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt)
        super().__init__(opt, shared)

    def _setup_data(self, path):
        super()._setup_data(path)
        if not self.opt['recovery']:
            for i, ep in enumerate(self.episodes):
                # make the signaling msg the label and remove the recovery msg
                texts = ep[0]['text'].split('\n')
                self.episodes[i][0].force_set('text', '\n'.join(texts[:-1]))
                self.episodes[i][0].force_set('labels', [texts[-1]])


class SaferDialoguesBADTeacher(ParlAIDialogTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('SaFeRDialogues options')
        agent.add_argument(
            '--recovery',
            type=bool,
            default=True,
            help="Whether or not to include the recovery utterance",
        )
        return parser

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _bad_path(opt)
        super().__init__(opt, shared)

    def _setup_data(self, path):
        super()._setup_data(path)
        if not self.opt['recovery']:
            for i, ep in enumerate(self.episodes):
                # make the signaling msg the label and remove the recovery msg
                texts = ep[0]['text'].split('\n')
                self.episodes[i][0].force_set('text', '\n'.join(texts[:-1]))
                self.episodes[i][0].force_set('labels', [texts[-1]])


class DefaultTeacher(SaferDialoguesTeacher):
    pass

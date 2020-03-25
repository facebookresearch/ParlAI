#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os

from parlai.core.teachers import DialogTeacher
from parlai.tasks.multinli.agents import setup_data, BICLASS_LABELS

from .build import build

SNLI = 'SNLI'
SNLI_VERSION = '1.0'
SNLI_PREFIX = 'snli_'


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        suffix = 'train'
    elif dt == 'valid':
        suffix = 'dev'
    elif dt == 'test':
        suffix = 'test'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(
        opt['datapath'],
        SNLI,
        SNLI_PREFIX + SNLI_VERSION,
        SNLI_PREFIX + SNLI_VERSION + '_' + suffix + '.jsonl',
    )

    return data_path


class DefaultTeacher(DialogTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('SNLI Teacher Args')
        parser.add_argument(
            '--dialog-format',
            type='bool',
            default=False,
            help="True if one would like to convert to a dialogue format without special tokens such as 'Premise'"
            " and 'Hypothesis' (default: False).",
        )

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'SNLI'
        self.dialog_format = opt.get('dialog_format', False)
        super().__init__(opt, shared)

    def setup_data(self, path):
        return setup_data(path, self.dialog_format)

    def label_candidates(self):
        if self.dialog_format:
            return BICLASS_LABELS
        return ('entailment', 'contradiction', 'neutral')

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os

from parlai.utils.io import PathManager
from parlai.core.teachers import DialogTeacher

from .build import build, DECODE, DECODE_PREFIX, DECODE_FOLDER_VERSION


def _path(opt, test_type):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        suffix = 'train'
    elif dt == 'valid':
        suffix = 'dev'
    elif dt == 'test':
        if test_type == 'vanilla':
            suffix = "test"
        else:
            suffix = test_type  # human-bot, act, or rct
    else:
        raise RuntimeError('Not valid datatype.')

    data_directory = os.path.join(
        opt['datapath'], DECODE, DECODE_PREFIX + DECODE_FOLDER_VERSION
    )

    data_path = os.path.join(data_directory, suffix + '.jsonl')

    return data_path


class DecodeTeacher(DialogTeacher):
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt):
        super().add_cmdline_args(parser, partial_opt)
        parser = parser.add_argument_group('DECODE Teacher Args')
        parser.add_argument(
            '--test_type',
            type=str,
            default='vanilla',
            help="The test sets can have three types: vanilla (default), human-bot, a2t, and rct.",
        )
        return parser

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        self.datatype = opt['datatype']
        test_type = opt.get('test_type', "vanilla")

        data_path = _path(opt, test_type)
        opt['datafile'] = data_path

        self.id = 'decode'
        # this can pass in other arguments.
        # self.dialog_format = opt.get('dialog_format', False)
        super().__init__(opt, shared)

    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            for data_line in data_file:
                data_item = json.loads(data_line)

                for turn_id, turn in enumerate(data_item['turns']):
                    new_episode = False
                    turn['labels'] = "none"
                    del turn['auxiliary']
                    if turn_id == 0:
                        new_episode = True
                    if turn_id == len(data_item['turns']) - 1:
                        if data_item['is_contradiction']:
                            turn['labels'] = 'contradiction'
                        else:
                            turn['labels'] = 'non_contradiction'
                        turn['auxiliary'] = {
                            "contradiction_indices": data_item[
                                'aggregated_contradiction_indices'
                            ]
                        }

                    yield turn, new_episode


class DefaultTeacher(DecodeTeacher):
    pass

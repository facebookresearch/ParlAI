#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
from typing import Optional
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.dialogue_safety.agents import StandardTeacher, NOT_OK_CLASS, OK_CLASS
from parlai.utils.io import PathManager
from .build import build, FILE_TYPE_EXTENSIONS, TROLL_TYPES
import os


def _path(opt):
    # build the data if it does not exist
    build(opt['datapath'])
    # set up path to data (specific to each dataset)
    data_path = os.path.join(opt['datapath'], 'safety_mix')
    return data_path


class SafetyMixTeacher(FixedDialogTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = StandardTeacher.add_cmdline_args(parser, partial_opt=partial_opt)
        parser = parser.add_argument_group('SafetyMix arguments')
        parser.add_argument(
            '--mix-user-type',
            type=str,
            default='troll',
            help=f'The troll user type you want in the safety mix. Possible options are: {TROLL_TYPES}',
        )
        return parser

    def __init__(self, opt, shared=None):
        self.opt = opt
        dpath = _path(opt)
        self.data_path = dpath
        self.troll_type = opt['mix_user_type']
        assert self.troll_type in TROLL_TYPES, f'{self.troll_type} not supported.'

        self.fixed_random = random.Random(42)
        self.label_candidates = [NOT_OK_CLASS, OK_CLASS]

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self._setup_data(opt['datatype'])
        super().__init__(opt, shared)
        self.reset()

    def _load_data_dump(self, datatype):
        d_type = datatype.split(':')[0]
        loaded_data = []
        with PathManager.open(
            os.path.join(
                self.data_path, f'{self.troll_type}{FILE_TYPE_EXTENSIONS[d_type]}'
            ),
            'rb',
        ) as f:
            dump = list(f)
        for json_str in dump:
            loaded_data.append(json.loads(json_str))
        return loaded_data

    def _setup_data(self, datatype):
        # load data
        self.data_dump = self._load_data_dump(datatype)
        self.data = self.data_dump
        self.fixed_random.shuffle(self.data)

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def get(self, episode_idx, entry_idx):
        return Message(self.data[episode_idx])

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared


class PosSafetyMixTeacher(SafetyMixTeacher):
    def _load_data_dump(self, datatype):
        d_type = datatype.split(':')[0]
        loaded_data = []
        with PathManager.open(
            os.path.join(
                self.data_path, f'pos_{self.troll_type}{FILE_TYPE_EXTENSIONS[d_type]}'
            ),
            'rb',
        ) as f:
            dump = list(f)
        for json_str in dump:
            loaded_data.append(json.loads(json_str))
        return loaded_data


class NegSafetyMixTeacher(SafetyMixTeacher):
    def _load_data_dump(self, datatype):
        d_type = datatype.split(':')[0]
        loaded_data = []
        with PathManager.open(
            os.path.join(
                self.data_path, f'neg_{self.troll_type}{FILE_TYPE_EXTENSIONS[d_type]}'
            ),
            'rb',
        ) as f:
            dump = list(f)
        for json_str in dump:
            loaded_data.append(json.loads(json_str))
        return loaded_data


class DefaultTeacher(SafetyMixTeacher):
    pass

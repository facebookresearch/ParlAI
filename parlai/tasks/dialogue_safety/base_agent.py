#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
import os
import random

from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher

from .build import build

SINGLE_TURN_DATA = 'single_turn_safety.json'


OK_CLASS = '__ok__'
NOT_OK_CLASS = '__notok__'


class _BaseSafetyTeacher(FixedDialogTeacher, ABC):
    """
    Abstract parent class for single turn safety teachers.

    Not meant to be a standalone teacher.
    """

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('Safety Teacher Args')
        parser.add_argument(
            '--round',
            type=int,
            default=1,
            choices=[1, 2, 3],
            help='Which round of data to use',
        )
        parser.add_argument(
            '--round-only',
            type='bool',
            default=False,
            help='if False, includes all rounds up to including the specified '
            'round; if True, only includes data from the specified round',
        )
        return parser

    def __init__(self, opt, shared=None):
        build(opt['datapath'])  # download the data
        self.opt = opt
        self.data_path = os.path.join(
            opt['datapath'], 'dialogue_safety', SINGLE_TURN_DATA
        )

        self.fixed_random = random.Random(42)
        self.round = opt['round']
        self.label_candidates = [NOT_OK_CLASS, OK_CLASS]

        if shared and 'data' in shared:
            self.data = shared['data']
        else:
            self._setup_data(opt['datatype'])

        super().__init__(opt, shared)
        self.reset()

    @abstractmethod
    def _load_data_dump(self):
        pass

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def _setup_data(self, datatype):
        # load data
        self.data_dump = self._load_data_dump()
        d_type = datatype.split(':')[0]
        self.data = []
        if not self.opt['round_only']:
            # loop and add other rounds
            for i in range(self.round - 1):
                rnd = str(i + 1)
                for x in ['good', 'bad']:
                    self.data += self.data_dump[d_type][rnd][x]

        # add data from current round
        for x in ['good', 'bad']:
            self.data += self.data_dump[d_type][str(self.round)][x]

        self.fixed_random.shuffle(self.data)

    def get(self, episode_idx, entry_idx):
        return Message(self.data[episode_idx])

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared

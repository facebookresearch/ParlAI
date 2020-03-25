#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os

from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from parlai.tasks.multinli.agents import (
    convert_to_dialogData,
    BICLASS_LABELS,
    MULTINLI_LABELS,
)

from .build import build

ANLI = 'ANLI'
ANLI_PREFIX = 'anli_'
ANLI_VERSION = 'v0.1'
ANLI_LABEL_DICT = {'e': 'entailment', 'c': 'contradiction', 'n': 'neutral'}
ANLI_PREMISE_KEY = 'context'
ANLI_HYPO_KEY = 'hypothesis'
ANLI_ANSWER_KEY = 'label'
ANLI_ROUNDS = ['R1', 'R2', 'R3']


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
        ANLI,
        ANLI_PREFIX + ANLI_VERSION,
        opt['round'],
        suffix + '.jsonl',
    )

    return data_path


class RoundBaseTeacher(DialogTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('RoundBase Teacher Args')
        parser.add_argument(
            '--to-parlaitext',
            type='bool',
            default=False,
            help="True if one would like to convert to 'ParlAI Text' format (default: False)",
        )

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['round'] = (
            opt['task'].split(':')[1] if len(opt['task'].split(':')) > 1 else 1
        )
        opt['round'] = opt['round'].upper()
        if not opt['round'] in ANLI_ROUNDS:
            raise KeyError(f"Undefined task round: {opt['round']}.")

        data_path = _path(opt)
        opt['datafile'] = data_path
        self.to_parlaitext = opt.get('to_parlaitext', False)
        self.id = opt['task'].upper()
        super().__init__(opt, shared)

    def label_candidates(self):
        if self.to_parlaitext:
            return BICLASS_LABELS
        return MULTINLI_LABELS

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path, 'r') as data_file:
            for pair_line in data_file:
                pair = json.loads(pair_line)
                if pair[ANLI_ANSWER_KEY] == '-':
                    continue

                label_raw = pair[ANLI_ANSWER_KEY]
                if label_raw in ANLI_LABEL_DICT:
                    label_raw = ANLI_LABEL_DICT[label_raw]

                question, answer, clas = convert_to_dialogData(
                    premise_raw=pair[ANLI_PREMISE_KEY],
                    hypo_raw=pair[ANLI_HYPO_KEY],
                    answer_raw=label_raw,
                    to_parlaitext=self.to_parlaitext,
                )

                yield (question, answer, None, clas), True


class R1Teacher(RoundBaseTeacher):
    pass


class R2Teacher(RoundBaseTeacher):
    pass


class R3Teacher(RoundBaseTeacher):
    pass


class DefaultTeacher(MultiTaskTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('ANLI Teacher Args')
        parser.add_argument(
            '--to-parlaitext',
            type='bool',
            default=False,
            help="True if one would like to convert to 'ParlAI Text' format (default: False)",
        )

    def __init__(self, opt, shared=None):
        anli_tasks = [
            'anli:r1',
            'anli:r2',
            'anli:r3',
        ]
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(anli_tasks)
        super().__init__(opt, shared)

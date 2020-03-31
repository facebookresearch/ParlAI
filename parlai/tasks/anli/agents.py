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


def _path(opt, anli_round):
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
        opt['datapath'], ANLI, ANLI_PREFIX + ANLI_VERSION, anli_round, suffix + '.jsonl'
    )

    return data_path


class RoundBaseTeacher(DialogTeacher):
    """
    Base class for teachers in all 3 rounds in ANLI tasks.

    ``RoundBaseTeacher`` derives anli_round (the round index of ANLI task which consists of 3 rounds NLI tasks with
    increasing difficulty. (See https://arxiv.org/abs/1910.14599 for more information)  from ``opt['task']``.
    ``anli_round`` is used to set the correct path to the downloaded data file for thaue specified round.

    ``RoundBaseTeacher`` also parses the requested dialog format(text format w/ or w/o special tokens 'Hypothesis',
    'Premise') and number of classes (3 classes entailment, neutral, contradiction or 2 classes contradiction, not
    contradiction) desired for particular training purposes.
    """

    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('RoundBase Teacher Args')
        parser.add_argument(
            '-dfm',
            '--dialog-format',
            type='bool',
            default=False,
            help="True if one would like to convert to a dialogue format without special tokens such as 'Premise'"
            " and 'Hypothesis' (default: False).",
        )
        parser.add_argument(
            '-bcl',
            '--binary-classes',
            type='bool',
            default=False,
            help="True if label candidates are (contradiction, not_contradiction), and (entailment, contradiction, "
            "neutral) otherwise (default: False).",
        )

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        self.anli_round = (
            opt['task'].split(':')[1] if len(opt['task'].split(':')) > 1 else None
        )
        self.anli_round = self.anli_round.upper()
        if self.anli_round not in ANLI_ROUNDS:
            raise KeyError(f"Undefined task round: {self.anli_round}.")

        data_path = _path(opt, self.anli_round)
        opt['datafile'] = data_path
        self.dialog_format = opt.get('dialog_format', False)
        self.binary_classes = opt.get('binary_classes', False)
        self.id = opt['task'].upper()
        super().__init__(opt, shared)

    def label_candidates(self):
        if self.binary_classes:
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

                question, answers, clas = convert_to_dialogData(
                    premise_raw=pair[ANLI_PREMISE_KEY],
                    hypo_raw=pair[ANLI_HYPO_KEY],
                    answer_raw=label_raw,
                    dialog_format=self.dialog_format,
                    binary_classes=self.binary_classes,
                )

                yield (question, answers, None, clas), True


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
            '-dfm',
            '--dialog-format',
            type='bool',
            default=False,
            help="True if one would like to convert to a dialogue format without special tokens such as 'Premise'"
            " and 'Hypothesis' (default: False).",
        ),
        parser.add_argument(
            '-bcl',
            '--binary-classes',
            type='bool',
            default=False,
            help="True if label candidates are (contradiction, not_contradiction), and (entailment, contradiction, "
            "neutral) otherwise (default: False).",
        )

    def __init__(self, opt, shared=None):
        anli_tasks = ['anli:r1', 'anli:r2', 'anli:r3']
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(anli_tasks)
        super().__init__(opt, shared)

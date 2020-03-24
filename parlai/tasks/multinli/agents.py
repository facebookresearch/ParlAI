#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build

import os
import copy
import json

MULTINLI = 'MultiNLI'
MULTINLI_VERSION = '1.0'
MULTINLI_PREFIX = 'multinli_'
MULTINLI_PREMISE_PREFIX = 'Premise: '
MULTINLI_HYPO_PREFIX = 'Hypothesis: '
MULTINLI_LABELS = ['entailment', 'contradiction', 'neutral']
MULTINLI_PREMISE_KEY = 'sentence1'
MULTINLI_HYPO_KEY = 'sentence2'
MULTINLI_ANSWER_KEY = 'gold_label'
NOT_CONTRADICT = 'not_contradiction'
BICLASS_LABELS = ['contradiction', NOT_CONTRADICT]


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        suffix = 'train'
    # Using matched set as valid and mismatched set as test
    elif dt == 'valid':
        suffix = 'dev_matched'
    elif dt == 'test':
        suffix = 'dev_mismatched'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(
        opt['datapath'],
        MULTINLI,
        MULTINLI_PREFIX + MULTINLI_VERSION,
        MULTINLI_PREFIX + MULTINLI_VERSION + '_' + suffix + '.jsonl',
    )
    return data_path


def setup_data(path, to_parlaitext=False):
    """
    Set up data in DialogData format from path
    :param path: path to the data file that stores the MNLI dataset
    :param to_parlaitext:
    :return:  a tuple in the parlai.core.teachers.DialogData form ``((x, y, r, c, i), new_episode?)`` where the ``x`` and ``new_episode``
        fields are mandatory and other fields may be omitted or ``None``.
    """
    print('loading: ' + path)

    with open(path, 'r') as data_file:
        for pair_line in data_file:
            pair = json.loads(pair_line)
            if pair[MULTINLI_ANSWER_KEY] == '-':
                continue

            question, answer, clas = convert_to_dialogData(premise_raw=pair[MULTINLI_PREMISE_KEY],
                                                           hypo_raw=pair[MULTINLI_HYPO_KEY],
                                                           answer_raw=pair[MULTINLI_ANSWER_KEY],
                                                           to_parlaitext=to_parlaitext)

            yield (question, answer, None, clas), True


def convert_to_dialogData(premise_raw, hypo_raw, answer_raw, to_parlaitext=False):
    """
    Convert from NLI context to dialog text
    :param premise_raw:
    :param hypo_raw:
    :param answer_raw:
    :param to_parlaitext:
    :return: a tuple (question, answer, clas)
        - ``question`` (str) is a query and possibly context
        - ``answer`` (iter) is an iterable of label(s) for that query
        - ``clas`` (iter) is an iterable of label candidates that the student can choose from
    """
    premise_raw = premise_raw.strip('\n').strip('\t')
    hypo_raw = hypo_raw.strip('\n').strip('\t')
    clas = MULTINLI_LABELS

    if to_parlaitext:
        if answer_raw != 'contradiction':
            answer_raw = NOT_CONTRADICT
        clas = BICLASS_LABELS
    else:
        premise_raw = MULTINLI_PREMISE_PREFIX + premise_raw
        hypo_raw = MULTINLI_HYPO_PREFIX + hypo_raw

    question = premise_raw + '\n' + hypo_raw
    answer = [answer_raw]

    return question, answer, clas


class DefaultTeacher(DialogTeacher):
    @staticmethod
    def add_cmdline_args(parser):
        parser = parser.add_argument_group('MNLI Teacher Args')
        parser.add_argument(
            '--to-parlaitext',
            type='bool',
            default=False,
            help="True if one would like to convert to 'Parlai Text' format (default: False)",
        )

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'MultiNLI'
        self.to_parlaitext = opt.get('to_parlaitext', False)
        super().__init__(opt, shared)

    def setup_data(self, path):
        return setup_data(path, self.to_parlaitext)

    def label_candidates(self):
        if self.to_parlaitext:
            return BICLASS_LABELS
        return MULTINLI_LABELS

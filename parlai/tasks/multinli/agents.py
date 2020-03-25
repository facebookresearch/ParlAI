#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os

from parlai.core.teachers import DialogTeacher

from .build import build

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


def setup_data(path, dialog_format=False):
    """
    Set up data in DialogData format from path.

    :param path: path to the data file that stores the MNLI dataset
    :param dialog_format: if set True, omit the special tokens 'Hypothesis' and 'Premise' in the text.
    :return:  a tuple in the parlai.core.teachers.DialogData format ``((x, y, r, c, i), new_episode?)`` where the ``x``
            is the query/question and ``y`` is the answer/label, ``clas`` represents the ``c`` the avaiable choices.
            ``new_episode`` is set True in any NLI teacher.
    """
    print('loading: ' + path)

    with open(path, 'r') as data_file:
        for pair_line in data_file:
            pair = json.loads(pair_line)
            if pair[MULTINLI_ANSWER_KEY] == '-':
                continue

            question, answer, clas = convert_to_dialogData(
                premise_raw=pair[MULTINLI_PREMISE_KEY],
                hypo_raw=pair[MULTINLI_HYPO_KEY],
                answer_raw=pair[MULTINLI_ANSWER_KEY],
                dialog_format=dialog_format,
            )

            yield (question, answer, None, clas), True


def convert_to_dialogData(premise_raw, hypo_raw, answer_raw, dialog_format=False):
    """
    Convert from NLI context to dialog text.

    :param premise_raw: raw premise extracted from jsonl file.
    :param hypo_raw: raw hypothesis extracted from jsonl file.
    :param answer_raw: raw answer extracted from jsonl file.
    :param dialog_format: if set True, omit the special tokens 'Hypothesis' and 'Premise' in the text.
    :return: a tuple (question, answer, clas)
        - ``question`` (str) is a query and possibly context
        - ``answer`` (iter) is an iterable of label(s) for that query
        - ``clas`` (iter) is an iterable of label candidates that the student can choose from
    """
    premise_raw = premise_raw.strip('\n').strip('\t')
    hypo_raw = hypo_raw.strip('\n').strip('\t')
    clas = MULTINLI_LABELS

    if dialog_format:
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
        self.id = 'MultiNLI'
        self.dialog_format = opt.get('dialog_format', False)
        super().__init__(opt, shared)

    def setup_data(self, path):
        return setup_data(path, self.dialog_format)

    def label_candidates(self):
        if self.dialog_format:
            return BICLASS_LABELS
        return MULTINLI_LABELS

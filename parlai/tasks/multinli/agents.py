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


def setup_data(path):
    print('loading: ' + path)

    with open(path, 'r') as data_file:
        for pair_line in data_file:
            pair = json.loads(pair_line)
            premise = MULTINLI_PREMISE_PREFIX + pair[MULTINLI_PREMISE_KEY]
            hypo = MULTINLI_HYPO_PREFIX + pair[MULTINLI_HYPO_KEY]
            answer = [pair[MULTINLI_ANSWER_KEY]]

            if answer == ['-']:
                continue

            question = premise + '\n' + hypo

            yield (question, answer, None, MULTINLI_LABELS), True


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'MultiNLI'
        super().__init__(opt, shared)

    def setup_data(self, path):
        return setup_data(path)

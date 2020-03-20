#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import json

from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from .build import build

from .anli_constants import ANLI, ANLI_ANSWER_KEY, ANLI_LABEL_DICT, ANLI_HYPO_KEY, \
    ANLI_HYPO_PREFIX, ANLI_LABELS, ANLI_PREFIX, ANLI_PREMISE_KEY, ANLI_PREMISE_PREFIX, ANLI_ROUNDS

ANLI_VERSION = 'v0.1'

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
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['round'] = opt['task'].split(':')[1] if len(opt['task'].split(':')) > 1 else 1
        opt['round'] = opt['round'].upper()

        if not opt['round'] in ANLI_ROUNDS:
            raise KeyError(f"Undefined task round: {opt['round']}.")

        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = opt['task'].upper()
        self.round = opt['round']
        super().__init__(opt, shared)

    def label_candidates(self):
        return ANLI_LABELS

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path, 'r') as data_file:
            for pair_line in data_file:
                pair = json.loads(pair_line)
                premise = ANLI_PREMISE_PREFIX + pair[ANLI_PREMISE_KEY]
                hypo = ANLI_HYPO_PREFIX + pair[ANLI_HYPO_KEY]
                answer = [ANLI_LABEL_DICT[pair[ANLI_ANSWER_KEY]]]

                if answer == ['-']:
                    continue

                question = premise + '\n' + hypo

                yield (question, answer, None, ANLI_LABELS), True


class R1Teacher(RoundBaseTeacher):
    pass


class R2Teacher(RoundBaseTeacher):
    pass


class R3Teacher(RoundBaseTeacher):
    pass


class DefaultTeacher(MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        anli_tasks = [
            'anli:r1',
            'anli:r2',
            'anli:r3',
        ]
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(anli_tasks)
        super().__init__(opt, shared)

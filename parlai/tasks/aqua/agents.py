#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build

import os
import copy
import json


AQUA = 'AQuA'
AQUA_QUESTION_KEY = 'question'
AQUA_ANSWER_KEY = 'correct'
AQUA_OPTIONS_KEY = 'options'
AQUA_RATIONALE_KEY = 'rationale'
RATIONALE_QUESTION_TEXT = 'Can you provide a rationale for your answer?'


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        prefix = 'train'
    # Using matched set as valid and mismatched set as test
    elif dt == 'valid':
        prefix = 'dev'
    elif dt == 'test':
        prefix = 'test'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], AQUA, AQUA, prefix + '.tok.json')

    return data_path


def setup_data(path):
    print('loading: ' + path)

    with open(path, 'r') as data_file:
        for line in data_file:
            question = json.loads(line)
            question_text = question[AQUA_QUESTION_KEY]
            answer = ord(question[AQUA_ANSWER_KEY]) - ord('A')
            labels = question[AQUA_OPTIONS_KEY]
            answer = [labels[answer]]
            yield (question_text, answer, None, labels), True

            # Ask for a rationale now
            rationale = [question[AQUA_RATIONALE_KEY]]
            yield (RATIONALE_QUESTION_TEXT, rationale, None, rationale), False


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'AQuA'

        super().__init__(opt, shared)

    def setup_data(self, path):
        return setup_data(path)

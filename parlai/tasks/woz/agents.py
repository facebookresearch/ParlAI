#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Download and build the data if it does not exist.

from parlai.core.teachers import DialogTeacher
from .build import build
import os
import json
import copy

FILE_START = 'woz_'
FILE_END = '_en.json'


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        suffix = 'train'
    # Using matched set as valid and mismatched set as test
    elif dt == 'valid':
        suffix = 'validate'
    elif dt == 'test':
        suffix = 'test'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'WoZ',
                             FILE_START + suffix + FILE_END)
    return data_path


class WoZTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):

        opt = copy.deepcopy(opt)
        data_path = _path(opt)

        opt['datafile'] = data_path

        # store datatype
        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'woz'

        build(opt)

        super().__init__(opt, shared)

    def setup_data(self, input_path):
        print('loading: ' + input_path)

        new_episode = True

        with open(input_path) as file:
            data = json.load(file)

        for dialogue in data:
            for line in dialogue['dialogue']:
                answer = [':'.join(turn_label) for turn_label in line['turn_label']]
                question = "What is the change in the dialogue state?"
                context = line['transcript']
                if answer:
                    yield (context + '\n' + question, answer, None, None), new_episode


class DefaultTeacher(WoZTeacher):
    pass

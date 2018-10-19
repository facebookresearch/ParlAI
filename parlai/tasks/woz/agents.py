#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Download and build the data if it does not exist.

from parlai.core.teachers import DialogTeacher
from .build import build
import os, json, copy

class WoZTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'woz'

        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'WoZ')
        self.opt = copy.deepcopy(opt)

        super().__init__(opt, shared)


    def setup_data (self, input_path):

        print('loading: ' + input_path)
        file_path = os.path.join(input_path, 'woz_train_en.json')

        new_episode = True

        with open(file_path) as file:
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


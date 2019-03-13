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


class MWSCTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.dt = opt['datatype'].split(':')[0]

        # store identifier for the teacher in the dialog
        self.id = 'mwsc'

        build(opt)
        self.datapath = os.path.join(opt['datapath'], 'MWSC')
        opt['datafile'] = self._path(opt)

        super().__init__(opt, shared)

    def _path(self, opt):
        build(opt)
        dt = opt['datatype'].split(':')[0]
        return os.path.join(self.datapath, dt + '.json')

    def setup_data(self, input_path):
        print('loading: ' + input_path)
        new_episode = True

        with open(input_path) as file:
            for l in file:
                schema_line = json.loads(l.strip())
                answer = schema_line.get('answer')
                question = schema_line.get('question')
                context = schema_line.get('context')
                yield (context + '\n' + question, [answer], None, None), new_episode


class DefaultTeacher(MWSCTeacher):
    pass

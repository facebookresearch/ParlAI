#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build

import json
import os
import glob


def _path(opt):
    build(opt)
    print('opt is', opt['datatype'])
    dt = opt['datatype'].split(':')[0]

    if dt == 'valid':
        dt = 'dev'
    elif dt != 'train' and dt != 'test':
        raise RuntimeError('Not valid datatype.')

    prefix = os.path.join(opt['datapath'], 'nlvr', 'nlvr-master', 'nlvr')
    questions_path = os.path.join(prefix, dt, dt + '.json')
    images_path = os.path.join(prefix, dt, 'images')

    return questions_path, images_path


class DefaultTeacher(DialogTeacher):
    # all possile answers for the questions
    cands = labels = ['true', 'false']

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        data_path, self.images_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'nlvr'
        self.dt = opt['datatype'].split(':')[0]
        if self.dt == 'valid':
            self.dt = 'dev'

        super().__init__(opt, shared)

    def label_candidates(self):
        return self.cands

    def setup_data(self, path):
        print('loading: ' + path)

        for line in open(path, 'r'):
            ques = json.loads(line)

            image_path = os.path.join(self.images_path, ques['directory'])
            image_file_names = glob.glob(
                image_path + '/' + self.dt + '-' + ques['identifier'] + '*'
            )

            question = "True or False: " + ques['sentence']
            answer = [ques['label']] if self.dt != 'test' else None
            # print( answer)
            yield (question, answer, None, None, image_file_names[0]), True

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build

import json
import os


def _path(opt):
    build(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'valid':
        dt = 'val'
    elif dt != 'train' and dt != 'test':
        raise RuntimeError('Not valid datatype.')

    prefix = os.path.join(opt['datapath'], 'CLEVR', 'CLEVR_v1.0')
    questions_path = os.path.join(
        prefix, 'questions', 'CLEVR_' + dt + '_questions.json'
    )
    images_path = os.path.join(prefix, 'images', dt)

    return questions_path, images_path


counts = [str(i) for i in range(11)]
materials = ['metal', 'rubber']
sizes = ['small', 'large']
shapes = ['cube', 'sphere', 'cylinder']
colors = ['gray', 'blue', 'brown', 'yellow', 'red', 'green', 'purple', 'cyan']


class DefaultTeacher(DialogTeacher):
    # all possile answers for the questions
    cands = ['yes', 'no'] + counts + materials + sizes + shapes + colors

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        data_path, self.images_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'clevr'

        super().__init__(opt, shared)

    def label_candidates(self):
        return self.cands

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            clevr = json.load(data_file)

        image_file = None
        for ques in clevr['questions']:
            # episode done if first question or image changed
            new_episode = ques['image_filename'] != image_file

            # only show image at beginning of episode
            image_file = ques['image_filename']
            img_path = None
            if new_episode:
                img_path = os.path.join(self.images_path, image_file)

            question = ques['question']
            answer = [ques['answer']] if ques['split'] != 'test' else None
            yield (question, answer, None, None, img_path), new_episode

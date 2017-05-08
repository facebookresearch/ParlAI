# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.dialog_teacher import DialogTeacher
from .build import build

from PIL import Image
import json
import random
import os

def _path(opt):
    build(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        suffix = 'train'
        img_suffix = os.path.join('train2014', 'COCO_train2014_')
    elif dt == 'valid':
        suffix = 'val'
        img_suffix = os.path.join('val2014', 'COCO_val2014_')
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'VisDial-v0.9',
        'visdial_0.9_' + suffix + '.json')

    image_path = os.path.join(opt['download_path'], img_suffix)

    return data_path, image_path


def _image_loader(path):
    """
    Loads the appropriate image from the image_id and returns PIL Image format.
    """
    return Image.open(path).convert('RGB')


class DefaultTeacher(DialogTeacher):
    """
    This version of VisDial inherits from the core Dialog Teacher, which just
    requires it to define an iterator over its data `setup_data` in order to
    inherit basic metrics, a `act` function, and enables
    Hogwild training with shared memory with no extra work.
    """
    def __init__(self, opt, shared=None):

        self.datatype = opt['datatype']
        data_path, image_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'visdial'

        super().__init__(opt, shared)

    def setup_data(self, path):
        print('loading: ' + path)
        with open(path) as data_file:
            self.visdial = json.load(data_file)

        self.questions = self.visdial['data']['questions']
        self.answers = self.visdial['data']['answers']

        for dialog in self.visdial['data']['dialogs']:
            # for each dialog
            image_id = dialog['dialog']
            caption = dialog['caption']
            episode_done = False
            for i, qa in enumerate(dialog['dialog']):
                if i == len(dialog['dialog']):
                    episode_done = True
                # for each question answer pair.
                question = self.questions[qa['question']]
                answer = [self.answers[qa['answer']]]
                answer_options = []
                for ans_id in qa['answer_options']:
                    answer_options.append(self.answers[ans_id])
                #answer_options = qa['answer_options']
                gt_index = qa['gt_index']
                yield (question, answer, 'None', answer_options), True

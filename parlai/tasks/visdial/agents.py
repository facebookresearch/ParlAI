#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import DialogTeacher
from .build import build
from parlai.tasks.coco_caption.build_2014 import buildImage

from PIL import Image
import json
import os


def _path(opt):
    build(opt)
    buildImage(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        suffix = 'train_train'
        img_suffix = os.path.join('train2014', 'COCO_train2014_')
    elif dt == 'valid':
        suffix = 'train_valid'
        img_suffix = os.path.join('train2014', 'COCO_train2014_')
    elif dt == 'test':
        suffix = 'val_test'
        img_suffix = os.path.join('val2014', 'COCO_val2014_')
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(
        opt['datapath'], 'VisDial-v0.9', 'visdial_0.9_' + suffix + '.json'
    )

    image_path = os.path.join(opt['datapath'], 'COCO-IMG-2014', img_suffix)

    return data_path, image_path


def _image_loader(path):
    """
    Loads the appropriate image from the image_id and returns PIL Image format.
    """
    return Image.open(path).convert('RGB')


class DefaultTeacher(DialogTeacher):
    """
    This version of VisDial inherits from the core Dialog Teacher, which just requires
    it to define an iterator over its data `setup_data` in order to inherit basic
    metrics, a `act` function, and enables Hogwild training with shared memory with no
    extra work.
    """

    def __init__(self, opt, shared=None):

        self.datatype = opt['datatype']
        data_path, self.image_path = _path(opt)
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
            image_id = dialog['image_id']
            caption = dialog['caption']
            img_path = self.image_path + '%012d.jpg' % (image_id)

            for i, qa in enumerate(dialog['dialog']):
                # for each question answer pair.
                question = self.questions[qa['question']]
                answer = [self.answers[qa['answer']]]
                answer_options = []
                for ans_id in qa['answer_options']:
                    answer_options.append(self.answers[ans_id])
                if i == 0:
                    # prepend with caption on first question
                    # only load image on first item
                    yield (
                        (
                            caption + '\n' + question,
                            answer,
                            None,
                            answer_options,
                            img_path,
                        ),
                        True,
                    )
                else:
                    yield (question, answer, None, answer_options), False

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Teacher
from .build import build, buildImage

from PIL import Image
import json
import random
import os


def _path(opt):
    build(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        ques_suffix = 'MultipleChoice_mscoco_train2014'
        annotation_suffix = 'mscoco_train2014'
        img_suffix = os.path.join('train2014', 'COCO_train2014_')
    elif dt == 'valid':
        ques_suffix = 'MultipleChoice_mscoco_val2014'
        annotation_suffix = 'mscoco_val2014'
        img_suffix = os.path.join('val2014', 'COCO_val2014_')
    else:
        ques_suffix = 'MultipleChoice_mscoco_test2015'
        annotation_suffix = 'None'
        img_suffix = os.path.join('test2014', 'COCO_test2014_')

    data_path = os.path.join(opt['datapath'], 'VQA-COCO2014',
        ques_suffix + '_questions.json')

    annotation_path = os.path.join(opt['datapath'], 'VQA-COCO2014',
        annotation_suffix + '_annotations.json')

    image_path = os.path.join(opt['datapath'], 'VQA-COCO2014', img_suffix)

    return data_path, annotation_path, image_path


def _image_loader(path):
    """
    Loads the appropriate image from the image_id and returns PIL Image format.
    """
    return Image.open(path).convert('RGB')


class OeTeacher(Teacher):
    """
    VQA Open-Ended teacher, which loads the json vqa data and implements its
    own `act` method for interacting with student agent.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.datatype = opt['datatype']
        data_path, annotation_path, image_path = _path(opt)
        self._setup_data(data_path, annotation_path, image_path)
        self.episode_idx = -1

    def __len__(self):
        return self.len

    def observe(self, observation):
        """Process observation for metrics. """
        if self.lastY is not None:
            loss = self.metrics.update(observation, self.lastY)
            self.lastY = None

    def act(self):
        if self.datatype == 'train':
            self.episode_idx = random.randrange(self.len)
        else:
            self.episode_idx = (self.episode_idx + 1) % self.len
            # always showing the same index now.
        qa = self.ques['questions'][self.episode_idx]
        question = qa['question']
        image_id = qa['image_id']

        img_path = self.image_path + '%012d.jpg' % (image_id)

        action = {
            'image': _image_loader(img_path),
            'text': question,
            'episode_done': True
        }

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][self.episode_idx]
            self.lastY = [ans['answer'] for ans in anno['answers']]

        if self.datatype.startswith('train'):
            action['labels'] = self.lastY

        return action

    def _setup_data(self, data_path, annotation_path, image_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.ques = json.load(data_file)

        if self.datatype != 'test':
            print('loading: ' + annotation_path)
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)

        self.image_path = image_path
        self.len = len(self.ques['questions'])


class McTeacher(OeTeacher):
    """
    VQA Multiple-Choice teacher, which inherits from OeTeacher but overrides
    the label and label_candidates fields with multiple choice data.
    """

    def act(self):
        action = super().act()

        qa = self.ques['questions'][self.episode_idx]
        multiple_choices = qa['multiple_choices']

        action['label_candidates'] = multiple_choices

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][self.episode_idx]
            self.lastY = [anno['multiple_choice_answer']]

        if self.datatype.startswith('train'):
            action['labels'] = self.lastY

        return action


class DefaultTeacher(McTeacher):
    # default to Multiple-Choice Teacher
    pass

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Teacher
from parlai.core.image_featurizers import ImageLoader
from .build import build, buildImage

import json
import random
import os


def _path(opt):
    build(opt)
    buildImage(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        ques_suffix = 'v2_OpenEnded_mscoco_train2014'
        annotation_suffix = 'v2_mscoco_train2014'
        img_suffix = os.path.join('train2014', 'COCO_train2014_')
    elif dt == 'valid':
        ques_suffix = 'v2_OpenEnded_mscoco_val2014'
        annotation_suffix = 'v2_mscoco_val2014'
        img_suffix = os.path.join('val2014', 'COCO_val2014_')
    elif dt == 'test':
        ques_suffix = 'v2_OpenEnded_mscoco_test2015'
        annotation_suffix = 'None'
        img_suffix = os.path.join('test2015', 'COCO_test2015_')
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'VQA-v2',
                             ques_suffix + '_questions.json')

    annotation_path = os.path.join(opt['datapath'], 'VQA-v2',
                                   annotation_suffix + '_annotations.json')

    image_path = os.path.join(opt['datapath'], 'COCO-IMG', img_suffix)

    return data_path, annotation_path, image_path


class OeTeacher(Teacher):
    """VQA v2.0 Open-Ended teacher, which loads the json VQA data and
    implements its own `act` method for interacting with student agent.
    agent.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt)
        self.datatype = opt['datatype']
        data_path, annotation_path, self.image_path = _path(opt)

        if shared and 'ques' in shared:
            self.ques = shared['ques']
            if 'annotation' in shared:
                self.annotation = shared['annotation']
        else:
            self._setup_data(data_path, annotation_path)
        self.len = len(self.ques['questions'])

        # for ordered data in batch mode (especially, for validation and
        # testing), each teacher in the batch gets a start index and a step
        # size so they all process disparate sets of the data
        self.step_size = opt.get('batchsize', 1)
        self.data_offset = opt.get('batchindex', 0)
        self.image_loader = ImageLoader(opt)

        self.reset()

    def __len__(self):
        return self.len

    def reset(self):
        # Reset the dialog so that it is at the start of the epoch,
        # and all metrics are reset.
        super().reset()
        self.lastY = None
        self.episode_idx = self.data_offset - self.step_size

    def observe(self, observation):
        """Process observation for metrics."""
        if self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    def act(self):
        if self.datatype == 'train':
            self.episode_idx = random.randrange(self.len)
        else:
            self.episode_idx = (self.episode_idx + self.step_size) % len(self)
            if self.episode_idx == len(self) - self.step_size:
                self.epochDone = True

        qa = self.ques['questions'][self.episode_idx]
        question = qa['question']
        image_id = qa['image_id']

        img_path = self.image_path + '%012d.jpg' % (image_id)

        action = {
            'image': self.image_loader.load(img_path),
            'text': question,
            'episode_done': True
        }

        if not self.datatype.startswith('test'):
            anno = self.annotation['annotations'][self.episode_idx]
            self.lastY = [ans['answer'] for ans in anno['answers']]

        if self.datatype.startswith('train'):
            action['labels'] = self.lastY

        return action

    def share(self):
        shared = super().share()
        shared['ques'] = self.ques
        if hasattr(self, 'annotation'):
            shared['annotation'] = self.annotation
        return shared

    def _setup_data(self, data_path, annotation_path):
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            self.ques = json.load(data_file)

        if self.datatype != 'test':
            print('loading: ' + annotation_path)
            with open(annotation_path) as data_file:
                self.annotation = json.load(data_file)


class DefaultTeacher(OeTeacher):
    pass

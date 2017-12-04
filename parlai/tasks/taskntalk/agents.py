# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.agents import Teacher
from .build import build

import json
import os
import random


def _path(opt, task_size='small'):
    # ensure data is built
    build(opt)
    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        file_name = 'train.json'
    elif dt == 'valid' or dt == 'test':
        file_name = 'valid.json'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'taskntalk', task_size, file_name)
    return data_path


class TaskNTalkTeacher(Teacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        # store datatype
        self.datatype = opt['datatype']
        if not shared:
            self._setup_data(self.opt['datafile'])
        else:
            self.data = shared['data']
            self.task_defn = shared['task_defn']
            self.task_index = shared['task_index']

    def _setup_data(self, data_path):
        # loads data
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            json_data = json.load(data_file)
            self.data = json_data['data']
            self.task_defn = json_data['task_defn']
        # images are [color, shape, style] lists (example: ['red', 'square', 'dotted'])
        self.task_index = {'color': 0, 'shape': 1, 'style': 2}
        random.shuffle(self.data)

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['task_defn'] = self.task_defn
        shared['task_index'] = self.task_index
        return shared

    def __len__(self):
        return len(self.data)

    def observe(self, observation):
        """Process observation for metrics."""
        self.observation = observation
        # TODO(kd): update metrics
        return observation

    def act(self):
        # TODO(kd): fetch all data for valid/test
        # select random image and task
        image = random.choice(self.data)
        task  = random.choice(self.task_defn)
        labels = [image[self.task_index[attr]] for attr in task]
        action = {
            'image': ' '.join(image),
            'text': ' '.join(task),
            'labels': [' '.join(labels)],
            'episode_done': True
        }
        return action


class SmallTeacher(TaskNTalkTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path(opt, 'small')
        super().__init__(opt, shared)


class LargeTeacher(TaskNTalkTeacher):
    def __init__(self, opt, shared=None):
        opt['datafile'] = _path(opt, 'large')
        super().__init__(opt, shared)


class DefaultTeacher(SmallTeacher):
    pass

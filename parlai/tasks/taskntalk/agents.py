#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import Teacher
from parlai.utils.io import PathManager
from .build import build

import json
import os
import random


def _path(opt, task_size='small'):
    """Return path to json file of dataset - it can be train/valid file
    of small/large dataset. Validation data is used for test as well,
    because labels are inferred from the image and task itself.
    """
    dt = opt['datatype'].split(':')[0]
    # ensure data is built
    build(opt)
    if dt == 'train':
        file_name = 'train.json'
    elif dt == 'valid' or dt == 'test':
        file_name = 'valid.json'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'taskntalk', task_size, file_name)
    return data_path


class TaskNTalkTeacher(Teacher):
    """
    TaskNTalk basic teacher, it picks a random image and associates a random task with
    it.

    Metric updates and observation are to be implemented.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.id = 'taskntalk'
        if not shared:
            self._setup_data(self.opt['datafile'])
        else:
            self.data = shared['data']
            self.task_defn = shared['task_defn']
            self.task_index = shared['task_index']

    def _setup_data(self, data_path):
        """
        Read the json file and store images and task definitions.
        """
        print('loading: ' + data_path)
        with PathManager.open(data_path) as data_file:
            json_data = json.load(data_file)
            self.data = json_data['data']
            self.task_defn = json_data['task_defn']
        # images are [color, shape, style] lists (example: ['red', 'square', 'dotted'])
        self.task_index = {'color': 0, 'shape': 1, 'style': 2}
        random.shuffle(self.data)

    def share(self):
        """
        Share images and task definitions with other teachers.
        """
        shared = super().share()
        shared['data'] = self.data
        shared['task_defn'] = self.task_defn
        shared['task_index'] = self.task_index
        return shared

    def __len__(self):
        return len(self.data)

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        self.observation = observation
        # TODO(kd): update metrics
        return observation

    def act(self):
        """
        Select random image and associate random task with it.
        """
        image = random.choice(self.data)
        task = random.choice(self.task_defn)
        labels = [image[self.task_index[attr]] for attr in task]
        action = {
            'image': ' '.join(image),
            'text': ' '.join(task),
            'labels': [' '.join(labels)],
            'episode_done': True,
        }
        # TODO(kd): fetch all data for valid/test
        return action


class SmallTeacher(TaskNTalkTeacher):
    """
    Teacher for small dataset, invoked by ``taskntalk:small``.
    """

    def __init__(self, opt, shared=None):
        opt['datafile'] = _path(opt, 'small')
        super().__init__(opt, shared)


class LargeTeacher(TaskNTalkTeacher):
    """
    Teacher for large dataset, invoked by ``taskntalk:large``.
    """

    def __init__(self, opt, shared=None):
        opt['datafile'] = _path(opt, 'large')
        super().__init__(opt, shared)


class DefaultTeacher(SmallTeacher):
    """
    Default teacher for small dataset, invoked by ``taskntalk``.
    """

    pass

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import itertools
import json
import os
import random

from parlai.core.agents import Teacher
from .build import build


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

        # task_size can be 'small' and 'large'
        self.task_size = opt.get('task', 'taskntalk:small')
        self.opt['datafile'] = _path(opt, self.task_size)

        # setup data if it hasn't been provided in shared
        if shared is not None:
            self._setup_data(data_path)
        else:
            self.data = shared['data']
            self.task_defn = shared['task_defn']
        self.batch_size = opt.get('batchsize', 1)

    def _setup_data(self, data_path):
        # loads data
        print('loading: ' + data_path)
        with open(data_path) as data_file:
            json_data = json.load(data_file)
            self.data = json_data['data']
            random.shuffle(self.data)
            self.task_defn = json_data['task_defn']

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['task_defn'] = self.task_defn
        return shared

    def __len__(self):
        return len(self.data)

    def observe(self, observation):
        """Process observation for metrics."""
        self.observation = observation
        # todo (karandesai) : update metrics
        return observation

    def act(self):
        # pick random example if training, else  fetch all data
        if self.datatype == 'train':
            # select random images
            indices = [random.randint(0, len(self.data) - 1)
                       for _ in range(self.batch_size)]
            images = [self.data[index] for index in indices]

            # select random tasks
            indices = [random.randint(0, len(self.task_defn) - 1)
                       for _ in range(self.batch_size)]
            tasks = [self.task_defn[index] for index in indices]
            action = {
                'image': images,
                'text': tasks,
                'episode_done': True
            }
        else:
            all_data = list(itertools.product(self.data, self.task_defn))
            action = {
                'image': [imtask_tuple[0] for imtask_tuple in all_data],
                'text': [imtask_tuple[1] for imtask_tuple in all_data],
                'episode_done': True
            }
        return action

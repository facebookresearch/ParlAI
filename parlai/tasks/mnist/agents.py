# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.agents import Teacher
from .build import build, data_fname

import os
import random
import pickle


def _path(opt):
    # Build the data if it doesn't exist
    build(opt)
    return os.path.join(opt['datapath'], 'mnist', data_fname)


class MnistTeacher(Teacher):
    """
    Mnist teacher, which loads the dataset and implements its
    own `act` method for interacting with student agent.
    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        # Fixed question and candidates for mnist task
        self.question = 'What number do you see?'
        self.candidates = [str(x) for x in range(10)]

        self.datatype = opt['datatype']
        data_path = _path(opt)

        if shared and 'data' in shared:
            self.data = shared['data']
            if 'labels' in shared:
                self.labels = shared['labels']
        else:
            self._setup_data(data_path)

        # for ordered data in batch mode (especially, for validation and
        # testing), each teacher in the batch gets a start index and a step
        # size so they all process disparate sets of the data
        self.step_size = opt.get('batchsize', 1)
        self.data_offset = opt.get('batchindex', 0)

        self.reset()

    def __len__(self):
        return len(self.data)

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
            self.episode_idx = random.randrange(len(self))
        else:
            self.episode_idx = (self.episode_idx + self.step_size) % len(self)
            if self.episode_idx == len(self) - self.step_size:
                self.epochDone = True

        action = {
            'image': self.data[self.episode_idx],
            'text': self.question,
            'episode_done': True,
            'label_candidates': self.candidates
        }

        if not self.datatype.startswith('test'):
            self.lastY = self.labels[self.episode_idx]

        if self.datatype.startswith('train'):
            action['labels'] = self.lastY

        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        if hasattr(self, 'labels'):
            shared['labels'] = self.labels
        return shared

    def _setup_data(self, data_path):
        print('loading: ' + data_path)

        if self.datatype == 'train':
            dt_idx = 0
        elif self.datatype == 'valid':
            dt_idx = 1
        elif self.datatype == 'test':
            dt_idx = 2
        else:
            raise RuntimeError('Not valid datatype.')

        with open(data_path, 'rb') as data_file:
            full_data = pickle.load(data_file)[dt_idx]
            self.data = full_data[0]
            self.labels = [str(y) for y in full_data[1]]


class DefaultTeacher(MnistTeacher):
    # default to Mnist Teacher
    pass

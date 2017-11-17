# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from .agents import Teacher

import random


class FixedDataTeacher(Teacher):
    """A teacher agent for all teachers involved in tasks with fixed data.

    This class provides the following functionality for its subclasses:

    - Resets a teacher
    - Provides an observe method
    - Computes and retrieves the next episode index for a teacher
    - TODO Provides a threadpool option for loading data (especially useful for
      large examples, e.g. images)

    """
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if not hasattr(self, 'datatype'):
            self.datatype = opt['datatype']
        if not hasattr(self, 'random'):
            self.random = self.datatype == 'train'
        if not hasattr(self, 'training'):
            self.training = self.datatype.startswith('train')
        # for ordered data in batch mode (especially, for validation and
        # testing), each teacher in the batch gets a start index and a step
        # size so they all process disparate sets of the data
        self.step_size = opt.get('batchsize', 1)
        self.data_offset = opt.get('batchindex', 0)
        self.reset()

    def __len__(self):
        return len(self.examples)

    def reset(self):
        """Reset the dialog so that it is at the start of the epoch,
        and all metrics are reset.
        """
        super().reset()
        self.metrics.clear()
        self.lastY = None
        self.episode_idx = self.data_offset - self.step_size
        self.episode_done = True
        self.epochDone = False
        try:
            if (self.episode_idx + self.step_size >= len(self) and not self.random):
                self.epochDone = True
        except AttributeError:
            # The data has not been initalized, so len(self) fails
            pass

    def observe(self, observation):
        """Process observation for metrics."""
        if self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    def next_episode_idx(self, num_eps=None):
        if not num_eps:
            num_eps = len(self)
        epoch_done = False
        if self.random:
            self.episode_idx = random.randrange(num_eps)
        else:
            self.episode_idx = (self.episode_idx + self.step_size) % num_eps
            if self.episode_idx + self.step_size >= num_eps:
                epoch_done = True
        return self.episode_idx, epoch_done

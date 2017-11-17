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
        if (not self.random and self.data_offset >= self.num_episodes()):
            self.epochDone = True

    def observe(self, observation):
        """Process observation for metrics."""
        if hasattr(self, 'lastY') and self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    def next_episode_idx(self, num_eps=None):
        if not num_eps:
            num_eps = self.num_episodes()
        if self.random:
            self.episode_idx = random.randrange(num_eps)
        else:
            self.episode_idx = (self.episode_idx + self.step_size) % num_eps
        return self.episode_idx

    def next_example(self):
        if self.episode_done:
            self.episode_idx = self.next_episode_idx()
            self.entry_idx = 0
        else:
            self.entry_idx += 1

        ex = self.get(self.episode_idx, self.entry_idx)
        self.episode_done = ex['episode_done']
        epoch_done = False

        if (not self.random and self.episode_done
                and self.episode_idx + self.step_size >= self.num_episodes()):
            epoch_done = True

        return ex, epoch_done

    def num_episodes(self):
        """Get the number of episodes in this dataset."""
        try:
            return len(self.episodes)
        except Exception:
            raise RuntimeError('"num_episodes" must be overriden by children.')

    def get(self, episode_idx, entry_idx=0):
        """Get the specified episode and the specified entry in that episode.

        Many datasets have only single-entry episodes, so entry_idx defaults to
        zero. Children must override this method in order to inherit the
        `next_example` method.
        """
        try:
            return self.examples[episode_idx][entry_idx]
        except Exception:
            raise RuntimeError('"Get" method must be overriden by children.')

    def act(self):
        """Send new dialog message."""
        if not hasattr(self, 'epochDone'):
            self.reset()
        if self.epochDone and not self.training:
            # need to call "reset" to repeat valid or test examples
            return {'episode_done': True, 'id': self.getID()}
        action, self.epochDone = self.next_example()
        action['id'] = self.getID()
        self.lastY = action.get('labels', None)
        if not self.datatype.startswith('train') and 'labels' in action:
            # move labels to eval field so not used for training
            # but this way the model can use the labels for perplexity or loss
            action['eval_labels'] = action.pop('labels')
        return action

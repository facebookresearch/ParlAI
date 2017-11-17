# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from .agents import Teacher

import concurrent.futures
from threading import Thread
import queue
import random

class DataLoader(Thread):
    """A worker thread that provides a threadpool for data loading.

    A teacher may submit a request to the loader, which will return the
    appropriate data.

    To submit a request, a teacher must provide the following:
        - a receive function (for receiving the data)
        - a load function (for loading the data), e.g. ImageLoader.load
        - args for the load function, e.g. an image path for an image loader
            -> args can be either a dictionary of arguments for a function, or
               a list of positional arguments
    """
    def __init__(self, opt):
        Thread.__init__(self, daemon=True)
        self.num_workers = opt.get('numthreads', 1)
        self.request_queue = queue.Queue()

    def __len__(self):
        return len(self.ques['questions'])

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            while True:
                receive_fn, load_fn, args = self.request_queue.get()
                if type(args) == dict:
                    future = executor.submit(load_fn, **args)
                else:
                    future = executor.submit(load_fn, *args)
                receive_fn(future)


class FixedDataTeacher(Teacher):
    """A teacher agent for all teachers involved in tasks with fixed data.

    This class provides the following functionality for its subclasses:

    - Resets a teacher
    - Provides an observe method
    - Computes and retrieves the next episode index for a teacher
    - Provides a threadpool option for loading data (especially useful for
      large data, e.g. images)

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

        self.data_queue = queue.Queue()
        if shared:
            self.data_loader = shared['data_loader']
        else:
            self.data_loader = DataLoader(opt)
            self.data_loader.start()

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
        self.data_queue = queue.Queue()
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

    def submit_data_request(self, receive_fn, load_fn, args):
        self.data_loader.request_queue.put((receive_fn, load_fn, args))

    def receive_data(self, future):
        data = future.result()
        self.data_queue.put(data)

    def share(self):
        shared = super().share()
        shared['data_loader'] = self.data_loader
        return shared

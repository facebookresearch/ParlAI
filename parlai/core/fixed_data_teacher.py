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

    To submit a request, a teacher should call ``request_load`` with the
    following arguments:
        - ``receive_fn`` - a receive function (for receiving the data)
        - ``load_fn`` - a load function (for loading the data)
        - ``args`` - arguments for the load function
            -> args can be either a dictionary of arguments for a function, or
               a list of positional arguments
    """
    def __init__(self, opt):
        Thread.__init__(self, daemon=True)
        self.num_workers = opt.get('numthreads', 1)
        self.request_queue = queue.Queue()

    def __len__(self):
        return len(self.ques['questions'])

    def request_load(self, receive_fn, load_fn, args):
        self.request_queue.put((receive_fn, load_fn, args))

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

    To utilize the DataLoader for threadpool loading, a teacher should
    implement the ``submit_load_request`` function to send a load request
    to the DataLoader by calling ``self.data_loader.request_load`` with the
    appropriate arguments (``receive_fn, load_fn, args``). The DataLoader then
    returns the data to the teacher's ``data_queue``, which the teacher can
    poll in its ``act`` method.

    The following is an example of the DataLoader usage in the VQA-V1 teacher.

        1. In the teacher's ``init`` function, the teacher calls its
           ``submit_load_request`` function to preload an image.
        2. The ``submit_load_request`` function gets the next ``episode_idx``,
           and computes the image path for the load request.
        3. At the end of ``submit_load_request``, the teacher calls
           ``self.data_loader.request_load`` with three args:
           - ``self.receive_data`` - the function that the DataLoader calls to
               return the the loaded object
           - ``self.image_loader.load`` - the function used to load the image
               from the image path
           - ``[img_path]`` - a list of arguments for the load function, which
               in this case is the path of the image.
         4. In the teacher's ``act`` function, the teacher loads the data from
            its data queue.
         5. At the end of the ``act`` function, the teacher calls
            ``submit_load_request`` to preload an image for the next example.


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
        if (not self.random and self.data_offset >= self.num_episodes()):
            self.epochDone = True

    def observe(self, observation):
        """Process observation for metrics."""
        if hasattr(self, 'lastY') and self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    def submit_load_request(self):
        """An agent should implement this method to submit requests to the
        data loader. At the end of this method, the agent should call
        ``self.data_loader.request_load()`` with the appropriate args.
        """
        pass

    def receive_data(self, future):
        """Function for receiving data from the data loader."""
        data = future.result()
        self.data_queue.put(data)

    def share(self):
        shared = super().share()
        shared['data_loader'] = self.data_loader
        return shared

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

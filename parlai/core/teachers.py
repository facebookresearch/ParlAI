#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This module provides a set of teachers that deal with dialog.

    ``FixedDialogTeacher(Teacher)``
    Base class for teachers in tasks that have fixed dialog - i.e., dialog
    that is not dynamically generated but rather is pulled from set examples.
    However, the class can be extended to all tasks involved fixed data.
    Implements much of the basic functionality of these teachers, including
    ``observe()``, ``act()``, ``next_example()``

    ``DialogTeacher(FixedDialogTeacher)``
     Base teacher class for doing dialog specifically with fixed chat logs.

    ``FbDialogTeacher(DialogTeacher)``
     Teacher class that provides access to data in the Facebook Dialog format.
     See the class description for more details.
     ** NOTE: ** We plan to deprecate this method soon in favor of ParlAIDialogTeacher,
     however several existing tasks are currently still using it.

    ``ParlAIDialogTeacher(DialogTeacher)``
     Teacher class that provides access to data in the ParlAI Dialog format.
     See the class description for more details.

This module also includes ``DataLoader``, a threadpool data loader for
``FixedDialogTeacher``, and ``DialogData``/``StreamDialogData``, data
structures for accessing textual dialog data and utilized by ``DialogTeacher``
"""

from parlai.core.agents import Teacher
from parlai.core.image_featurizers import ImageLoader
from parlai.core.message import Message
from parlai.utils.misc import AttrDict, no_lock, str_to_msg, warn_once

from functools import lru_cache
from abc import abstractmethod

import concurrent.futures
import multiprocessing
from multiprocessing import Value, Lock
from threading import Thread
import queue
import random
import sys
import time
import os
import torch
import json
import argparse


class DataLoader(Thread):
    """
    A worker thread that provides a threadpool for data loading.

    A teacher may submit a request to the loader, which will return the
    appropriate data.

    To submit a request, a teacher should call ``request_load``.
    """

    def __init__(self, opt):
        Thread.__init__(self, daemon=True)
        self.num_workers = opt.get('num_load_threads', 1)
        self.request_queue = queue.Queue()

    def request_load(self, receive_fn, load_fn, args):
        """
        Queue a request for loading.

        :param receive_fn:
            a receive function (for receiving the data)
        :param load_fn:
            a load function (for loading the data)
        :param args:
            arguments for the load function. args can be either a dictionary of
            arguments for a function, or a list of positional arguments
        """
        self.request_queue.put((receive_fn, load_fn, args))

    def run(self):
        """
        Run the execution loop.
        """
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers)
        with executor:
            while True:
                receive_fn, load_fn, args = self.request_queue.get()
                if type(args) == dict:
                    future = executor.submit(load_fn, **args)
                else:
                    future = executor.submit(load_fn, *args)
                receive_fn(future)


class FixedDialogTeacher(Teacher):
    """
    A teacher agent for all teachers involved in tasks with fixed data.

    This class provides the following functionality for its subclasses:

    - Resets a teacher
    - Provides an observe method
    - Computes and retrieves the next episode index for a teacher
    - Provides a threadpool option for loading data (especially useful for
      large data, e.g. images)

    In order to take advantage of the first few features, all a subclass has to
    implement is three functions: ``num_episodes``, ``num_examples``, and
    ``get`` (which returns a specific example from a specific episode).

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

    To see this in action, take a look at this teacher in ``tasks.vqa_v1.agents``.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)

        if not hasattr(self, 'datatype'):
            self.datatype = opt['datatype']
        if not hasattr(self, 'random'):
            self.random = self.datatype == 'train'
        if not hasattr(self, 'training'):
            self.training = self.datatype.startswith('train')
        if not hasattr(self, 'datafile'):
            self.datafile = opt.get('datafile', opt.get('pytorch_datafile'))
        # set up support for multithreaded data loading
        self.data_queue = queue.Queue()
        if shared:
            self.index = shared['index']
            if 'data_loader' in shared:
                self.data_loader = shared['data_loader']
            if 'threadindex' in shared:
                self.threadindex = shared['threadindex']
            if 'examples' in shared:
                self.examples = shared['examples']
        else:
            self.index = AttrDict(value=-1)

        if not hasattr(self, 'data_loader'):
            self.data_loader = DataLoader(opt)
            self.data_loader.start()

        # set up batching
        self.bsz = opt.get('batchsize', 1)
        self.batchindex = opt.get('batchindex', 0)
        self.use_batch_act = False  # Batch act disabled by default

    def _lock(self):
        if hasattr(self.index, 'get_lock'):
            return self.index.get_lock()
        else:
            return no_lock()

    def reset(self):
        """
        Reset the dialog to the start of the epoch, and reset all metrics.
        """
        super().reset()
        self.metrics.clear()
        self.lastY = None
        self.episode_done = True
        self.epochDone = False
        self.data_queue = queue.Queue()

        self.episode_idx = -1
        with self._lock():
            self.index.value = -1

    def submit_load_request(self):
        """
        Submit a load request.

        An agent should implement this method to submit requests to the data
        loader. At the end of this method, the agent should call
        ``self.data_loader.request_load()`` with the appropriate args.

        By default, this method does nothing.
        """
        # TODO: mark as abstract
        pass

    def receive_data(self, future):
        """
        Receive data from the data loader.

        :param future: result from the load request.
        """
        data = future.result()
        self.data_queue.put(data)

    def share(self):
        """
        Share the data and dataloader.
        """
        shared = super().share()

        if hasattr(self, 'lastYs'):
            # share lastYs to communicate between batch_act and observe
            shared['lastYs'] = self.lastYs

        if hasattr(self, 'examples'):
            shared['examples'] = self.examples

        if hasattr(self, 'data_loader'):
            shared['data_loader'] = self.data_loader

        if self.opt.get('numthreads', 1) > 1:
            if type(self.index) is not multiprocessing.sharedctypes.Synchronized:
                # for multithreading need to move index into threadsafe memory
                self.index = Value('l', -1)

        shared['index'] = self.index

        return shared

    def next_episode_idx(self, num_eps=None, loop=None):
        """
        Return the next episode index.

        :param num_eps:
            default None uses ``num_episodes`` value.
        :param loop:
            default None loops during training but not evaluation.
        """
        if num_eps is None:
            num_eps = self.num_episodes()
        if loop is None:
            loop = self.training
        if self.random:
            new_idx = random.randrange(num_eps)
        else:
            with self._lock():
                self.index.value += 1
                if loop:
                    self.index.value %= num_eps
                new_idx = self.index.value
        return new_idx

    def next_example(self):
        """
        Return the next example.

        If there are multiple examples in the same episode, returns the next one in that
        episode. If that episode is over, gets a new episode index and returns the first
        example of that episode.
        """
        if self.episode_done:
            self.episode_idx = self.next_episode_idx()
            self.entry_idx = 0
        else:
            self.entry_idx += 1

        if self.episode_idx >= self.num_episodes():
            return {'episode_done': True}, True

        ex = self.get(self.episode_idx, self.entry_idx)
        self.episode_done = ex.get('episode_done', False)

        if (
            not self.random
            and self.episode_done
            and self.episode_idx + self.opt.get("batchsize", 1) >= self.num_episodes()
        ):
            epoch_done = True
        else:
            epoch_done = False

        return ex, epoch_done

    def next_batch(self):
        """
        Return the next batch of examples.
        """
        # get next batch
        with self._lock():
            self.index.value += 1
            if self.training:
                self.index.value %= len(self.batches)
            batch_idx = self.index.value

            if batch_idx + 1 >= len(self.batches):
                if self.random:
                    random.shuffle(self.batches)
                self.epochDone = True
            else:
                self.epochDone = False

        if batch_idx >= len(self.batches):
            return [{'episode_done': True, 'id': self.getID()}] * self.bsz

        return self.batches[batch_idx]

    def num_episodes(self):
        """
        Get the number of episodes in this dataset.
        """
        if self.use_batch_act:
            # when using batch_act, this is length of sorted data
            return len(self.sorted_data)
        raise RuntimeError('"num_episodes" must be overriden by children.')

    @lru_cache(maxsize=1)
    def num_examples(self):
        """
        Get the total number of examples in this dataset.
        """
        if self.use_batch_act:
            # when using batch_act, this is length of sorted data
            return len(self.sorted_data)
        raise RuntimeError('"num_examples" must be overriden by children.')

    def get(self, episode_idx, entry_idx=0):
        """
        Get the specified episode and the specified entry in that episode.

        Children must override this method in order to inherit the
        `next_example` method.

        :param episode_idx:
            which episode to return examples from
        :param entry_idx:
            which example to return from the episode.  Many datasets have only
            single-entry episodes, so this defaults to zero.
        """
        # TODO: mark as abstract, get rid of runtime error.
        raise RuntimeError('"Get" method must be overriden by children.')

    def observe(self, observation):
        """
        Process observation for metrics.
        """
        if self.use_batch_act:
            self.lastY = self.lastYs[self.batchindex]
            self.lastYs[self.batchindex] = None

        if hasattr(self, 'lastY') and self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    def batch_act(self, observations):
        """
        Return an entire batch of examples.

        Note: Currently used by PytorchDataTeacher.
        """
        # DEPRECATEDAY: looks like this isn't used anymore
        # we ignore observations
        if not hasattr(self, 'epochDone'):
            # reset if haven't yet
            self.reset()

        batch = self.next_batch()
        # pad batch
        if len(batch) < self.bsz:
            batch += [{'episode_done': True, 'id': self.getID()}] * (
                self.bsz - len(batch)
            )

        # remember correct answer if available (for padding, None)
        for i, ex in enumerate(batch):
            if 'labels' in ex:
                labels = ex['labels']
                self.lastYs[i] = labels
                if not self.datatype.startswith('train') or 'evalmode' in self.datatype:
                    del ex['labels']
                    if not self.opt.get('hide_labels', False):
                        ex['eval_labels'] = labels
            else:
                self.lastYs[i] = ex.get('eval_labels', None)

        return batch

    def act(self):
        """
        Send new dialog message.
        """
        if not hasattr(self, 'epochDone'):
            # reset if haven't yet
            self.reset()

        # get next example, action is episode_done dict if already out of exs
        action, self.epochDone = self.next_example()
        # TODO: all teachers should eventually create messages
        # while setting up the data, so this won't be necessary
        action = Message(action)
        action.force_set('id', self.getID())

        # remember correct answer if available
        self.lastY = action.get('labels', action.get('eval_labels', None))
        if (
            not self.datatype.startswith('train') or 'evalmode' in self.datatype
        ) and 'labels' in action:
            # move labels to eval field so not used for training
            # but this way the model can use the labels for perplexity or loss
            action = action.copy()
            labels = action.pop('labels')
            if not self.opt.get('hide_labels', False):
                action['eval_labels'] = labels

        return action


class DialogTeacher(FixedDialogTeacher):
    """
    A base teacher class for doing dialog with fixed chat logs.

    This class provides a set a basic functionality:

    - uses data class to store and query text data
    - generates action tables to send to the student agent from the data

    If you have ``opt.numthreads > 1``, this also activates a shared memory
    array for the data and lock-protected shared-memory metrics.

    In order to subclass this class, you must implement ``setup_data()`` in
    your class (or subclass another class which does, like
    ``FbDialogTeacher``), which reads your data file as an iterator.
    """

    def __init__(self, opt, shared=None):
        # Check for setup_data
        if not hasattr(self, 'setup_data'):
            raise RuntimeError(
                'Must implement setup_data or subclass a class '
                'which implements it (e.g. FbDialogTeacher) '
                'in order to use this class.'
            )
        super().__init__(opt, shared)

        self.startTime = time.time()
        self.datatype = opt['datatype']
        self.training = self.datatype.startswith('train')
        self.stream = 'stream' in self.datatype.split(':')

        if not self.use_batch_act:
            # first initialize any shared objects
            data_class = StreamDialogData if self.stream else DialogData
            kwargs = {'cycle': self.training} if self.stream else {}
            if shared and shared.get('data'):
                self.data = data_class(opt, shared=shared['data'], **kwargs)
            else:
                self.data = data_class(
                    opt,
                    data_loader=self.setup_data,
                    cands=self.label_candidates(),
                    **kwargs,
                )

        self.reset()

    def reset(self):
        """
        Reset the dialog to the start of the epoch, reset all metrics.
        """
        super().reset()
        if self.stream:
            self.data.reset()
            self.epochDone = False

    def share(self):
        """
        Share the data.
        """
        shared = super().share()
        if hasattr(self, 'data'):
            shared['data'] = self.data.share()
        return shared

    def label_candidates(self):
        """
        Provide consistent label candidates for all examples.

        Default implementation returns ``None`` always, but this may be overriden to
        provide candidates in all areas. See ``FbDialogueTeacher``.
        """
        # TODO DEPRECATIONDAY: FbiDialogueTeacher is being deprecated, should we
        # remove this?

        # TODO: mark as optionally abstract?
        return None

    def num_episodes(self):
        """
        Return the number of episodes in the data.
        """
        try:
            return self.data.num_episodes()
        except AttributeError:
            return super().num_episodes()

    @lru_cache(maxsize=1)
    def num_examples(self):
        """
        Return the number of examples in the data.
        """
        try:
            return self.data.num_examples()
        except AttributeError:
            return super().num_examples()

    def get(self, episode_idx, entry_idx=0):
        """
        Get a specific example.
        """
        return self.data.get(episode_idx, entry_idx)[0]

    def next_example(self):
        """
        Get the next example.
        """
        if self.stream:
            action, epoch_done = self.data.get()
        else:
            action, epoch_done = super().next_example()
        return action, epoch_done


class DialogData(object):
    """
    Provides a data structure for accessing textual dialog data.

    This can be used whenever the dialog data is a fixed log of chats
    (i.e not a simulator setting). The logs can include dialog text and possibly
    supervised labels, candidate labels and rewards.

    All these are stored in this internal data format which is used by the
    ``DialogTeacher`` class.

    :param opt:
        options to initialize the class
    :param data_loader:
        an iterable with each call returning a tuple in the form
        ``((x, y, r, c, i), new_episode?)`` where the ``x`` and ``new_episode``
        fields are mandatory and other fields may be omitted or ``None``.
    :param cands:
        can be set to provide a list of candidate labels for every example in
        this dataset, which the agent can choose from (the correct answer
        should be in this set).

    :param random:
        tells the data class whether or not to visit episodes sequentially or
        randomly when returning examples to the caller.

    The contents of the ``((x, y, r, c, i), new_episode?)`` tuples returned by
    the data loader is the following:

    - ``x`` (str) is a query and possibly context
    - ``y`` (iter) is an iterable of label(s) for that query
    - ``r`` (str) is the str reward for getting that query correct
    - ``c`` (iter) is an iterable of label candidates that the student can choose from
    - ``i`` (str) is a str path to an image on disk, which will be loaded by the
      data class at request-time. should always point to the raw image file.
    - ``new_episode?`` (bool) is a boolean value specifying whether that example
      is the start of a new episode. If you don't use episodes set this
      to ``True`` every time.
    """

    def __init__(self, opt, data_loader=None, cands=None, shared=None, **kwargs):
        # self.data is a list of episodes
        # each episode is a tuple of entries
        # each entry is a tuple of values for the action/observation table
        if shared:
            self.image_loader = shared.get('image_loader', None)
            self.data = shared.get('data', [])
            self.cands = shared.get('cands', None)
        else:
            self.image_loader = ImageLoader(opt)
            self.data = []
            self._load(data_loader, opt['datafile'])
            self.cands = None if cands is None else set(sys.intern(c) for c in cands)
        self.addedCands = []
        self.copied_cands = False

    def share(self):
        """
        Share the data.
        """
        shared = {
            'data': self.data,
            'cands': self.cands,
            'image_loader': self.image_loader,
        }
        return shared

    def _read_episode(self, data_loader):
        """
        Read one episode at a time from the provided iterable over entries.

        :param data_loader:
            an iterable which returns tuples in the format described in the
            class docstring.
        """
        episode = []
        last_cands = None
        for entry, new in data_loader:
            if new and len(episode) > 0:
                yield tuple(episode)
                episode = []
                last_cands = None

            # intern all strings so we don't store them more than once
            # TODO: clean up the if .. sys.intern else None by refactoring
            new_entry = []
            if len(entry) > 0:
                # process text if available
                if entry[0] is not None:
                    new_entry.append(sys.intern(entry[0]))
                else:
                    new_entry.append(None)
                # TODO: unindent all of these one level.
                if len(entry) > 1:
                    # process labels if available
                    if entry[1] is None:
                        new_entry.append(None)
                    elif hasattr(entry[1], '__iter__') and type(entry[1]) is not str:
                        # TODO: this could use the abc collections
                        # make sure iterable over labels, not single string
                        new_entry.append(tuple(sys.intern(e) for e in entry[1]))
                    else:
                        raise TypeError(
                            'Must provide iterable over labels, not a single string.'
                        )
                if len(entry) > 2:
                    # process reward if available
                    if entry[2] is not None:
                        new_entry.append(entry[2])
                    else:
                        new_entry.append(None)
                if len(entry) > 3:
                    # process label candidates if available
                    if entry[3] is None:
                        new_entry.append(None)
                    elif last_cands and entry[3] is last_cands:
                        # if cands are shared, say "same" so we
                        # don't store them again
                        # TODO: This is bad, and it's not actually used anywhere
                        # DEPRECATIONDAY: make this more rational
                        new_entry.append(sys.intern('same as last time'))
                    elif hasattr(entry[3], '__iter__') and type(entry[3]) is not str:
                        # make sure iterable over candidates, not single string
                        last_cands = entry[3]
                        new_entry.append(tuple(sys.intern(e) for e in entry[3]))
                    else:
                        raise TypeError(
                            'Must provide iterable over label candidates, '
                            'not a single string.'
                        )
                if len(entry) > 4 and entry[4] is not None:
                    new_entry.append(sys.intern(entry[4]))

            episode.append(tuple(new_entry))

        if len(episode) > 0:
            yield tuple(episode)

    def _load(self, data_loader, datafile):
        """
        Load up data from an iterable over tuples described in the class docs.

        :param iter data_loader:
            an iterator which returns tuples in the format described in the
            class docstring.
        :param str datafile:
        """
        for episode in self._read_episode(data_loader(datafile)):
            self.data.append(episode)

    def num_episodes(self):
        """
        Return number of episodes in the dataset.
        """
        return len(self.data)

    @lru_cache(maxsize=1)
    def num_examples(self):
        """
        Return total number of entries available.

        Each episode has at least one entry, but might have many more.
        """
        return sum(len(episode) for episode in self.data)

    def get(self, episode_idx, entry_idx=0):
        """
        Get the specified episode and the specified entry in that episode.

        :param episode_idx:
            which episode to return examples from
        :param entry_idx:
            which example to return from the episode. Many datasets have only
            single-entry episodes, so this defaults to zero.
        """
        # first look up data
        episode = self.data[episode_idx]
        entry = episode[entry_idx]
        episode_done = entry_idx == len(episode) - 1
        end_of_data = episode_done and episode_idx == len(self.data) - 1

        # now pack it in a action-observation dictionary
        table = self.build_table(entry)

        # last entry in this episode
        table['episode_done'] = episode_done
        return table, end_of_data

    def build_table(self, entry):
        """
        Packs an entry into an action-observation dictionary.

        :param entry: a tuple in the form described in the class docstring.
        """
        table = {}
        if entry[0] is not None:
            table['text'] = entry[0]
        if len(entry) > 1:
            if entry[1] is not None:
                table['labels'] = entry[1]
        if len(entry) > 2:
            if entry[2] is not None:
                table['reward'] = entry[2]
        if len(entry) > 3:
            if entry[3] is not None:
                table['label_candidates'] = entry[3]
        if len(entry) > 4 and entry[4] is not None:
            img = self.image_loader.load(entry[4])
            if img is not None:
                table['image'] = img

        if table.get('labels', None) is not None and self.cands is not None:
            if self.addedCands:
                # remove elements in addedCands
                self.cands.difference_update(self.addedCands)
                self.addedCands.clear()
            for label in table['labels']:
                if label not in self.cands:
                    # add labels, queue them for removal next time
                    if not self.copied_cands:
                        self.cands = self.cands.copy()
                        self.copied_cands = True
                    self.cands.add(label)
                    self.addedCands.append(label)
            table['label_candidates'] = self.cands

        if 'labels' in table and 'label_candidates' in table:
            if table['labels'][0] not in table['label_candidates']:
                raise RuntimeError('true label missing from candidate labels')
        return table


class StreamDialogData(DialogData):
    """
    Provides a data structure for streaming textual dialog data.

    This can be used whenever the dialog data follows the format described in
    DialogData but cannot fit entirely into memory.

    Additional keyword-argument cycle defines if the stream should restart from
    the beginning after an epoch is finished (defaults to True).

    :param opt:
        options to initialize the class
    :param data_loader:
        an iterable with each call returning a tuple in the form
        ``((x, y, r, c, i), new_episode?)`` where the ``x`` and ``new_episode``
        fields are mandatory and other fields may be omitted or ``None``.
    :param cands:
        can be set to provide a list of candidate labels for every example in
        this dataset, which the agent can choose from (the correct answer
        should be in this set).
    :param random:
        tells the data class whether or not to visit episodes sequentially or
        randomly when returning examples to the caller.
    :param cycle:
        (default True) whether to restart at beginning when end of stream
        reached without reset being called.
    """

    def __init__(self, opt, data_loader=None, cands=None, shared=None, **kwargs):
        # super() call initiates stream in self.data by calling _load()
        super().__init__(opt, data_loader, cands, shared, **kwargs)
        self.cycle = kwargs['cycle'] if 'cycle' in kwargs else True
        if shared:
            # auxiliary instances hold pointer to main datastream in self.data
            self.reset_data = shared['reset']
            # Share datafile and data_loader for computing num_exs and num_eps
            self.datafile = shared['datafile']
            self.data_loader = shared['data_loader']
            if 'lock' in shared:
                self.lock = shared['lock']
        else:
            # main instance holds the stream and shares pointer to it
            self.data_loader = data_loader
            self.datafile = opt['datafile']
            self.reset_data = None
            self.is_reset = True
            if opt.get('numthreads', 1) > 1:
                print(
                    'WARNING: multithreaded streaming will process every '
                    'example numthreads times.'
                )
                self.lock = Lock()
        self.entry_idx = 0
        self.next_episode = None
        self.num_eps = None
        self.num_exs = None

    def share(self):
        """
        Share the stream.
        """
        shared = super().share()
        # also share reset method to allow datastream to be reset
        shared['reset'] = self.reset
        # share datafile and data for loading length if necessary
        shared['datafile'] = self.datafile
        shared['data_loader'] = self.data_loader
        if hasattr(self, 'lock'):
            shared['lock'] = self.lock

        return shared

    def _load(self, data_loader, datafile):
        """
        Load data generator into data field.
        """
        self.data = self._data_generator(data_loader, datafile)

    def _data_generator(self, data_loader, datafile):
        """
        Generate data using the iterator over tuples constructed by data_loader.
        """
        self.is_reset = False
        while True:
            for episode in self._read_episode(data_loader(datafile)):
                yield episode
            yield -1
            while not self.cycle:
                yield -1

    def load_length(self):
        """
        Calculate the length of the dataset and caches it in a file.

        Note that this can take some time for large datasets. Episode and entry indexes
        cannot be specified during streaming.
        """
        datafiles = self.datafile if type(self.datafile) is tuple else [self.datafile]
        length_file = datafiles[0] + ".lengths"
        if not os.path.isfile(length_file):
            num_eps = 0
            num_exs = 0
            for episode in self._read_episode(self.data_loader(self.datafile)):
                num_eps += 1
                num_exs += len(episode)
            with open(length_file, 'w') as f:
                f.write("{}\n{}".format(num_eps, num_exs))
        else:
            with open(length_file, 'r') as f:
                num_eps, num_exs = f.readlines()
        return int(num_eps), int(num_exs)

    def num_examples(self):
        """
        Return the number of examples in the data.
        """
        if not self.num_exs:
            self.num_eps, self.num_exs = self.load_length()
        return self.num_exs

    def num_episodes(self):
        """
        Return the number of episodes in the data.
        """
        if not self.num_eps:
            self.num_eps, self.num_exs = self.load_length()
        return self.num_eps

    def _lock(self):
        if hasattr(self, 'lock'):
            return self.lock
        else:
            return no_lock()

    def get(self):
        """
        Get the next entry from the stream.

        When episode is done returns first entry of next episode.
        """
        # first look up data
        if self.next_episode != -1 or self.entry_idx != 0:
            with self._lock():
                if self.next_episode is None:
                    self.next_episode = next(self.data)
                if self.entry_idx == 0:
                    self.cur_episode = self.next_episode
                    self.next_episode = next(self.data)
                entry = self.cur_episode[self.entry_idx]

                # now pack it in a action-observation dictionary
                table = self.build_table(entry)

                episode_done = self.entry_idx == len(self.cur_episode) - 1
                if episode_done:
                    self.entry_idx = 0
                else:
                    self.entry_idx += 1
                end_of_data = episode_done and self.next_episode is -1
                if end_of_data and self.cycle:
                    self.next_episode = next(self.data)

                # last entry in this episode
                table['episode_done'] = episode_done
        else:
            table = {'episode_done': True}
            end_of_data = True

        return table, end_of_data

    def reset(self):
        """
        Reset the datastream to its beginning.
        """
        if self.reset_data is not None:
            # auxiliary instance, reset main datastream
            self.data = self.reset_data()
            self.next_episode = None
        elif not self.is_reset:
            # if main instance is not reset, reset datastream
            self._load(self.data_loader, self.datafile)
            self.is_reset = True
            self.next_episode = None
        self.entry_idx = 0
        return self.data


class FbDialogTeacher(DialogTeacher):
    """
    This module provides access to data in the Facebook Dialog format.

    Subclasses ``DialogTeacher`` for functionality and provides an
    implementation of ``setup_data()`` which iterates over datasets in the
    "fbdialog" format. If your data is in the format below, use this class to
    handle file parsing for you.

    The way FB Dialog data is set up is as follows:

    ::

        1 Sam went to the kitchen.
        2 Pat gave Sam the milk.
        3 Where is the milk?<TAB>kitchen<TAB>1<TAB>hallway|kitchen|bathroom
        4 Sam went to the hallway.
        5 Pat went to the bathroom.
        6 Where is the milk?<TAB>hallway<TAB>1<TAB>hallway|kitchen|bathroom

    Lines 1-6 represent a single episode, with two different examples: the
    first example is lines 1-3, and the second is lines 4-6.

    Lines 1,2,4, and 5 represent contextual information.

    Lines 3 and 6 contain a query, a label, a reward for getting the question
    correct, and three label candidates.

    Since both of these examples are part of the same episode, the information
    provided in the first example is relevant to the query in the second
    example and therefore the agent must remember the first example in order to
    do well.

    In general dialog in this format can contain any speech, not just QA pairs:

    ::

        1 Hi how's it going?<TAB>It's going great. What's new?
        2 Well I'm working on a new project at work.<TAB>Oh me too!
        3 Oh cool!<TAB>Tell me about yours.

    etc.

    Note that dialogs are interpreted as being one-way. For example, consider
    this dialog:

    ::

        1 X1    Y1
        2 X2    Y2
        3 X3    Y3

    A set of examples X1 => Y1, X2 => Y2, and X3 => Y3 will be generated.
    However, Y1 => X2 and Y2 => X3 are not created as separate examples by
    default. This makes sense for some data (we don't need to train on the idea
    that "kitchen" should be followed by "Sam went to the hallway..." above),
    but for other datasets it may be helpful to add additional examples in the
    reverse direction ("Oh cool!" is a response to "Oh me too!" above).
    """

    def __init__(self, opt, shared=None):
        self.opt = opt
        self.cloze = opt.get('cloze', False)
        if shared and 'cands' in shared:
            self.cands = shared['cands']
        else:
            self.cands = self.load_cands(opt.get('cands_datafile', None))
        super().__init__(opt, shared)

    def share(self):
        """
        Share the data and canidates.
        """
        shared = super().share()
        shared['cands'] = self.cands
        return shared

    def label_candidates(self):
        """
        Return the candidates.
        """
        return self.cands

    def load_cands(self, path):
        """
        Load a global fixed set of candidates.

        The candidates will be provided by the teacher for every example (the true
        labels for a specific example are also added to this set, so that it's possible
        to get the right answer).
        """
        if path is None:
            return None
        cands = []
        lines_have_ids = False
        cands_are_replies = False
        cnt = 0
        with open(path) as read:
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) > 0:
                    cnt = cnt + 1
                    # If lines are numbered we strip them of numbers.
                    if cnt == 1 and line[0:2] == '1 ':
                        lines_have_ids = True
                    # If tabs then the label_candidates are all the replies.
                    if '\t' in line and not cands_are_replies:
                        cands_are_replies = True
                        cands = []
                    if lines_have_ids:
                        space_idx = line.find(' ')
                        line = line[space_idx + 1 :]
                        if cands_are_replies:
                            sp = line.split('\t')
                            if len(sp) > 1 and sp[1] != '':
                                cands.append(sp[1])
                        else:
                            cands.append(line)
                    else:
                        cands.append(line)
        return cands

    def setup_data(self, path):
        r"""
        Read data in the fbdialog format.

        Returns ``((x,y,r,c), new_episode?)`` tuples.

        ``x`` represents a query, ``y`` represents the labels, ``r`` represents
        any reward, and ``c`` represents any label_candidates.

        The example above will be translated into the following tuples:

        ::

            x: 'Sam went to the kitchen\nPat gave Sam the milk\nWhere is the milk?'
            y: ['kitchen']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = True (this is the first example in the episode)


        ::

            x: 'Sam went to the hallway\\nPat went to the bathroom\\nWhere is the
                milk?'
            y: ['hallway']
            r: '1'
            c: ['hallway', 'kitchen', 'bathroom']
            new_episode = False (this is the second example in the episode)
        """
        print("[loading fbdialog data:" + path + "]")
        with open(path) as read:
            start = True
            x = ''
            reward = 0
            last_conv_id = None
            for line in read:
                line = line.strip().replace('\\n', '\n')
                if len(line) == 0:
                    # empty response
                    continue

                # first, get conversation index -- '1' means start of episode
                space_idx = line.find(' ')
                if space_idx == -1:
                    # empty line, both individuals are saying whitespace
                    conv_id = int(line)
                else:
                    conv_id = int(line[:space_idx])

                # split line into constituent parts, if available:
                # x<tab>y<tab>reward<tab>label_candidates
                # where y, reward, and label_candidates are optional
                split = line[space_idx + 1 :].split('\t')

                # remove empty items and strip each one
                for i in range(len(split)):
                    word = split[i].strip()
                    if len(word) == 0:
                        split[i] = ''
                    else:
                        split[i] = word
                # Empty reward string same as None
                if len(split) > 2 and split[2] == '':
                    split[2] = None

                # now check if we're at a new episode
                if last_conv_id is None or conv_id <= last_conv_id:
                    x = x.strip()
                    if x:
                        yield [x, None, reward], start
                    start = True
                    reward = 0
                    # start a new episode
                    if self.cloze:
                        x = 'Fill in the blank in the last sentence.\n{x}'.format(
                            x=split[0]
                        )
                    else:
                        x = split[0]
                else:
                    if x:
                        # otherwise add current x to what we have so far
                        x = '{x}\n{next_x}'.format(x=x, next_x=split[0])
                    else:
                        x = split[0]
                last_conv_id = conv_id
                if len(split) > 2 and split[2]:
                    reward += float(split[2])

                if len(split) > 1 and split[1]:
                    # only generate an example if we have a y
                    split[0] = x
                    # split labels
                    split[1] = split[1].split('|')
                    if len(split) > 3:
                        # split label_candidates
                        split[3] = split[3].split('|')
                    if len(split) > 2:
                        split[2] = reward
                    else:
                        split.append(reward)
                    if start:
                        yield split, True
                        start = False
                    else:
                        yield split, False
                    # reset x in case there is unlabeled data still left
                    x = ''
                    reward = 0
            if x:
                yield [x, None, reward], start


class ParlAIDialogTeacher(FixedDialogTeacher):
    """
    This module provides access to data in the ParlAI Text Dialog format.

    Subclasses ``FixedDialogTeacher`` for functionality and provides an
    implementation of ``setup_data()`` which iterates over datasets in the
    "ParlAI text" format. If your data is in the format below, use this class to
    handle file parsing for you.

    The way the data is set up is as follows:

    ::

        text:Sam went to the kitchen. <NEWL>
        Pat gave Sam the milk. <NEWL>
        Where is the milk? <TAB> labels:kitchen <TAB> reward:1
        <TAB> label_candidates:hallway|kitchen|bathroom
        text:Sam went to the hallway. <NEWL>
        Pat went to the bathroom. <NEWL>
        Where is the milk? <TAB> labels:hallway <TAB> reward:1
        <TAB> label_candidates:hallway|kitchen|bathroom <TAB> episode_done:True

    Lines 1-2 represent a single episode, with a different example on each line.
    The lines contain a query and a label for getting the question
    correct, and three label candidates.

    Since both of these examples are part of the same episode, the information
    provided in the first example is relevant to the query in the second
    example and therefore the agent must remember the first example in order to
    do well.

    In general dialog this format can contain any speech, not just QA pairs:

    ::

        text:Hi how's it going?<TAB>labels:It's going great. What's new?
        text:Well I'm working on a new project at work.<TAB>labels:Oh me too!
        text:Oh cool!<TAB>labels:Tell me about yours.

    etc.

    Note that dialogs are interpreted as being one-way. For example, consider
    this dialog:

    ::

        1 X1    Y1
        2 X2    Y2
        3 X3    Y3

    A set of examples X1 => Y1, X2 => Y2, and X3 => Y3 will be generated.
    However, Y1 => X2 and Y2 => X3 are not created as separate examples by
    default. This makes sense for some data (we don't need to train on the idea
    that "kitchen" should be followed by "Sam went to the hallway..." above),
    but for other datasets it may be helpful to add additional examples in the
    reverse direction ("Oh cool!" is a response to "Oh me too!" above).
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        if not shared:
            self.episodes = []
            self.num_exs = 0
            if opt.get('parlaidialogteacher_datafile') is not None:
                self._setup_data(opt.get('parlaidialogteacher_datafile'))
        else:
            self.episodes = shared['episodes']
            self.num_exs = sum(len(e) for e in self.episodes)
        self.id = opt.get('parlaidialogteacher_datafile', 'teacher')
        self.reset()

    def share(self):
        """
        Share the episodes.
        """
        shared = super().share()
        shared['episodes'] = self.episodes
        return shared

    def num_examples(self):
        """
        Return the number of examples from the data.
        """
        return self.num_exs

    def num_episodes(self):
        """
        Return the number of episodes from the data.
        """
        return len(self.episodes)

    def get(self, episode_idx, entry_idx=None):
        """
        Get a specific example from the dataset.
        """
        return self.episodes[episode_idx][entry_idx]

    def _setup_data(self, path):
        print("[loading parlAI text data:" + path + "]")
        self.episodes = []
        self.num_exs = 0
        eps = []
        with open(path, newline='\n') as read:
            for line in read:
                msg = str_to_msg(line.rstrip('\n'))
                if msg:
                    self.num_exs += 1
                    eps.append(msg)
                    if msg.get('episode_done', False):
                        self.episodes.append(eps)
                        eps = []
        if len(eps) > 0:
            # add last episode
            eps[-1].force_set('episode_done', True)
            self.episodes.append(eps)


class AbstractImageTeacher(FixedDialogTeacher):
    """
    Abstract class to allow easier creation of image + dialogue tasks.

    This class handles creating image features via ImageLoader if applicable
    (resnet, resnext variants) or loading existing image features from a dict
    path as per get_image_features_path().

    Important methods and properties (override in subclass if needed):
    - get_data_path(): where data file is found (default: <datapath>/<task>)
    - get_image_path(): where images found (default: <datapath>/<task>/images)
    - get_image_features_path(): dict of image features (default:
      <datapath>/<task>/image_features)
    - @property image_id_key: which key in data file objects represents image_id
    - @property text_key: which key in data file objects represents text
    Note: Assumes data files are named <dt>.json

    @abstractmethod image_id_to_image_path() must be implemented in
    subclass

    Example with the key defaults (but the keys can be customized):
    obs = {'text': <caption>,
        'image': <image features if specified else image>
        }
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        self.task = opt['task'].split(':')[1] if ':' in opt['task'] else opt['task']
        self.data_path = self.get_data_path(opt)
        self.data = self.load_data(self.data_path, self.opt)
        self.datatype = opt.get('datatype').split(':')[0]

        # Example of available models: 'resnet152', 'resnext101_32x48d_wsl',
        # and ImageLoader supports other resnet and resnext models too
        # Raises an Exception if not valid
        self._validate_image_mode_name(opt.get('image_mode'))

        # IMPORTANT NOTE: this teacher will be instantiated twice. The first
        # by build_dict in which case the image_mode is to 'no_image_model' to
        # avoid calculating image features twice.
        self.image_mode = opt.get('image_mode')

        # Not using default image_mode paramater b/c there is a normalization
        # (or bug) somewhere in build_dict that is setting it to none
        self.include_image = opt.get('image_mode') != 'no_image_model'

        self.image_path = self.get_image_path(opt)
        self.image_loader = None
        self.image_features_dict = self.get_image_features_path(
            self.task, self.image_mode, self.datatype
        )
        self.image_features_dim = opt.get('image_features_dim')
        self.blank_image_features = torch.FloatTensor(self.image_features_dim).fill_(0)

        if shared and 'data' in shared:
            self.data = shared['data']
            self.image_loader = shared['image_loader']
            if 'image_features_dict' in shared:
                self.image_features_dict = shared['image_features_dict']
        elif self.include_image:
            self.setup_image_features(self.data_path)
        else:
            # This will happen when building the dictionary - is normal
            # build_dict sets image_mode to 'none'
            warn_once('AbstractImageTeacher self.include_image was False')

        # TODO: won't need this after we have proper logging levels set
        self.__verbose = False

        self.reset()

    def get_available_image_mode_names(self):
        """
        Available image model names.

        resnet and resnext variants available from the ImageLoader. resnext101_XXXXX_wsl
        is the open-sourced FB AI model (960m images, 1.5k hashtags, finetuned on
        ImageNet).
        """
        available_model_names = ImageLoader.get_available_model_names()
        return ['no_image_model'] + available_model_names

    def _validate_image_mode_name(self, a):
        """
        Validate the image_mode passed in.

        Needed because image_mode used elsewhere in ParlAI is not always consistent with
        what the image teacher allows.
        """
        if not isinstance(a, str):
            raise argparse.ArgumentTypeError(
                '%s must be a string representing image model name' % a
            )
        available_model_names = self.get_available_image_mode_names()
        if a not in available_model_names:
            raise argparse.ArgumentTypeError(
                '\"%s\" unknown image model name. Choose from: %s. Currently suggested resnet is resnet152 and resnext is resnext101_32x48d_wsl.'
                % (a, available_model_names)
            )
        return a

    @classmethod
    def add_cmdline_args(cls, argparser):
        # Be sure to call super() if overriding this method b/c
        # AbstractImageTeacher has necessary params
        agent = argparser.add_argument_group('AbstractImageTeacher Arguments')
        agent.add_argument(
            '--image-path',
            type=str,
            default=None,
            help='Optional argument to specify where images for dataset are'
            'stored if already downloaded. Most tasks will download the images'
            'if not present on the < datapath > / < task > _images / * and * if'
            'this argument is not specified.',
        )

        agent.add_argument(
            '--image-features-dim',
            type=int,
            default=2048,
            help='Specify the size of image features Tensors.',
        )

    @property
    def image_id_key(self):
        """
        Which key in the input data dict objects uniquely identify each image.

        Common image keys are "image_id" or "image_num". May be implemented by subclass.
        """
        return 'image_id'

    @property
    def text_key(self):
        """
        Which key in the input data dict objects identifies the text.

        Common keys are "text" or "comment". May be implemented by subclass.
        """
        return 'text'

    @abstractmethod
    def image_id_to_image_path(self, image_id):
        """
        Get the path of the image on disk.

        Must be implemented by subclass.
        """
        pass

    def get_data_path(self, opt):
        """
        Determines path to the data file.
        """
        task_name = opt['task'].split(':')[1] if ':' in opt['task'] else opt['task']
        data_path = os.path.join(opt['datapath'], task_name)
        return data_path

    def get_image_path(self, opt):
        """
        Return the path to the data directory and to the image directory.

        Is based on opt fields: task, datatype (train, valid, test), datapath.

        Subclass can override this.
        """
        data_path = self.get_data_path(opt)
        if opt.get('image_path', None):
            image_path = opt['image_path']
        else:
            # other common choice: .join(opt['datapath'], task_name + '_images')
            image_path = os.path.join(data_path, 'images')

        return image_path

    def get_image_features_path(self, task, image_model_name, dt):
        """
        Image features for the dataset images are stored here.

        Can be overriden in subclass to use custom paths. Image features can be manually
        copied into this directory or in the case of ImageLoader eligible models, they
        will be built and stored here if not already there.
        """
        # In default implementation, self.data_path already has task name added
        image_features_path = os.path.join(self.data_path, 'image_features')

        if not os.path.isdir(image_features_path):
            os.makedirs(image_features_path)

        return os.path.join(
            image_features_path, '%s_%s_%s_features_dict' % (task, image_model_name, dt)
        )

    def is_image_mode_buildable(self, model_name):
        """
        Is buildable if features can be calculated by ImageLoader.

        Users may wish to compute features for the dataset offline and use in the model,
        in which case, the image model should return False and get_image_features()
        should be overriden in subclass.
        """
        return model_name in ImageLoader.get_available_model_names()

    def load_data(self, data_path, opt):
        """
        Loading the data file, which is the index to the images and text.

        It is often a .json file with the name of the <datatype>.json (i.e.
        train.json). Stores in self.data.

        Can be override by subclass.
        """

        dt = opt['datatype'].split(':')[0]

        # Sometimes file is named "val" instead of "valid"
        if dt not in ['train', 'valid', 'val', 'test']:
            raise Exception(
                'Unknown dt parameter: %s. Expected either "train", "valid", or "test".'
                % dt
            )

        # Assumes file is train.json or valid.json named
        data_file = os.path.join(self.data_path, '%s.json' % dt)

        # Load the text data and image number indexes
        with open(data_file) as f:
            self.data = json.load(f)

        if len(self.data) > 0 and self.image_id_key not in self.data[0]:
            # Data doesn't have a "image_id" like field so add the index in the file to the data
            for idx, d in enumerate(self.data):
                d[self.image_id_key] = idx

        return self.data

    def setup_image_features(self, data_path):
        """
        Load text and image data.

        The image features all live in dicts by default in <data_path>/
        image_features/ but get_image_features_path() above can be overriden by
        subclass to put them elsewhere.

        In the (very odd) case that the resnet or resnext dicts (models
        buildable using ImageLoader) are not found, we build them.
        """

        image_mode_features_dict_path = self.get_image_features_path(
            self.task, self.image_mode, self.datatype
        )

        if os.path.isfile(image_mode_features_dict_path):
            print(
                f'Loading existing image features dict for model: {self.image_mode} at: {image_mode_features_dict_path}'
            )
            self.image_features_dict = torch.load(
                image_mode_features_dict_path, map_location='cpu'
            )
        else:
            print(f'No existing image features, attempting to build.')
            if self.is_image_mode_buildable(self.image_mode):
                # TODO: Awkward to modify the input opt but needed to use
                # TODO: ImageLoader functionality. Is from comment_battle,
                # TODO: will refactor this at some point soon most likely
                image_loader_opt = self.opt.copy()
                image_loader_opt['image_mode'] = (
                    self.image_mode if self.include_image else 'no_image_model'
                )

                image_loader_opt['image_size'] = 256
                image_loader_opt['image_cropsize'] = 224
                self.image_loader = ImageLoader(image_loader_opt)

                # try to build with ImageLoader (i.e. resenet/resnext variants)
                self.image_features_dict = self._build_image_features_dict(
                    self.data_path, self.datatype, image_mode_features_dict_path
                )
            else:
                raise RuntimeError(
                    'Image model: %s is not buildable by ImageLoader but does'
                    'not already exist on disk as an image features dict for'
                    'this dataset.' % self.image_mode
                )

    def _build_image_features_dict(self, data_path, dt, store_dict_path):
        """
        Build resne(x)t image features with ImageLoader.

        (Or anything handleable by ImageLoader) and save to path. Only called if we
        haven't already built the dict before.
        """
        image_features_dict = {}
        total = len(self.data)
        import tqdm

        pbar = tqdm.tqdm(
            total=total,
            unit='cand',
            unit_scale=True,
            desc='Building image features dict for %s with ImageLoader.'
            % self.image_mode,
        )
        num = 0
        for ex in self.data:
            img_id = ex[self.image_id_key]
            img_path = self.image_id_to_image_path(img_id)
            image = self.image_loader.load(img_path).detach()
            image = image[0, :, 0, 0]
            image_features_dict[img_id] = image
            num += 1
            pbar.update(1)
            if num % 1000 == 0:
                print(f'Processing image index: {num}')
        torch.save(image_features_dict, store_dict_path)
        return image_features_dict

    def reset(self):
        super().reset()
        self.example = None

    def num_episodes(self):
        return self.num_examples()

    def num_examples(self):
        return len(self.data)

    def get_image_features(self, example):
        """
        Get image features for example.

        Can be overrided in subclass for different behavior. For large datasets, it may
        be more appropriate to use the ImageLoader.load() method to load image features
        (as this is essentially streaming the features from disk, so that we do not have
        to load a large image feature dict in memory). #TODO Could be the default option
        if we are using -dt train:stream
        """

        key = str(example[self.image_id_key])
        if not self.include_image or key not in self.image_features_dict:
            image_features = self.blank_image_features
        else:
            image_features = self.image_features_dict[key]
        return image_features

    def get(self, episode_idx, entry_idx=0):
        """
        Override this in subclass if your data should be handled in a different format.
        """
        example = self.data[episode_idx]
        image_features = self.get_image_features(example)
        return {
            'labels': [example[self.text_key]],
            'image': image_features,
            'episode_idx': episode_idx,
            'episode_done': True,
        }

    def next_example(self):
        """
        Load queued example.

        When self.image_features_dict is not `none`, this is essentially a no-op.
        However, this is necessary/useful if self.image_mode is e.g. raw or ascii.
        """
        return_result = None
        if self.include_image:
            if self.image_features_dict is not None:
                # We have specified an image model and it uses image features either
                # generated by ImageLoader or by pre-computation and manually
                # placed on path
                return_result = super().next_example()
            else:
                # We have specified an image model/mode but it's not based on
                # calculating features, (e.g. "raw" or "ascii") so we try to load
                # the image from the DataLoader
                if self.example is not None:
                    # Move the image we previously loaded in the background via
                    # the DataLoader into the example
                    image = self.data_queue.get()
                    self.example['image'] = image
                    return_result = self.example, self.imageEpochDone
                else:
                    return_result = super().next_example()

                # Now, get next base example and put it to load in the background
                self.example, self.imageEpochDone = super().next_example()
                img_path = self.image_id_to_image_path(self.example['image_id'])
                self.data_loader.request_load(
                    self.receive_data, self.image_loader.load, (img_path,)
                )
        else:
            # No image model specified
            return_result = super().next_example()
        if self.__verbose:
            print(f'AbstractImageTeacher - next_example(): {return_result}')
        return return_result

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        shared['image_loader'] = self.image_loader
        if hasattr(self, 'image_features_dict'):
            shared['image_features_dict'] = self.image_features_dict
        return shared

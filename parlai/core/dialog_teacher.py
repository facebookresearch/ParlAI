# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from .agents import Teacher

from .image_featurizers import ImageLoader

import logging
from multiprocessing import Queue, Value
import os
import random
import sys
import time


class DialogTeacher(Teacher):
    """A base teacher class for doing dialog with fixed chat logs.

    This class provides a set a basic functionality:

    - uses data class to store and query text data
    - generates action tables to send to the student agent from the data
    - metrics tracking count of sent vs correctly answered queries

    If you have ``opt.numthreads > 1``, this also activates a shared memory
    array for the data and lock-protected shared-memory metrics.

    In order to subclass this class, you must implement ``setup_data()`` in your
    class (or subclass another class which does, like ``FbDialogTeacher``), which
    reads your data file as an iterator.
    """

    def __init__(self, opt, shared=None):
        # Check for setup_data
        if not hasattr(self, 'setup_data'):
            raise RuntimeError('Must implement setup_data or subclass a class' +
                               ' which implements it (e.g. FbDialogTeacher)' +
                               ' in order to use this class.')

        super().__init__(opt, shared)

        self.datatype = opt['datatype']
        self.startTime = time.time()
        self.stream = 'stream' in opt['datatype'].split(':')

        # first initialize any shared objects
        self.random = self.datatype == 'train'
        data_class = StreamDialogData if self.stream else DialogData
        kwargs = {'cycle': 'train' in self.datatype} if self.stream else {}
        if shared and shared.get('data'):
            self.data = data_class(opt, shared=shared['data'], **kwargs)
        else:
            self.data = data_class(opt, data_loader=self.setup_data,
                    cands=self.label_candidates(), **kwargs)

        # for ordered data in batch mode (especially, for validation and
        # testing), each teacher in the batch gets a start index and a step
        # size so they all process disparate sets of the data
        self.step_size = opt.get('batchsize', 1)
        self.data_offset = opt.get('batchindex', 0)

        self.reset()

    def reset(self):
        # Reset the dialog so that it is at the start of the epoch,
        # and all metrics are reset.
        self.metrics.clear()
        self.lastY = None
        self.episode_idx = self.data_offset - self.step_size
        self.episode_done = True
        self.epochDone = False
        if self.stream:
            self.data.reset()
        elif not self.random and self.data_offset >= self.data.num_episodes():
            # could have bigger batchsize then episodes... so nothing to do
            self.epochDone = True

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.epochDone = False
        return self

    def __next__(self):
        if self.epochDone:
            raise StopIteration()

    def share(self):
        shared = super().share()
        shared['data'] = self.data.share()
        return shared

    def label_candidates(self):
        """Returns ``None`` by default, but override this in children (such as
        ``FbDialogTeacher``) to load up candidate labels for every example.
        """
        return None

    def observe(self, observation):
        """Process observation for metrics. """
        if self.lastY is not None:
            self.metrics.update(observation, self.lastY)
            self.lastY = None
        return observation

    def next_example(self):
        num_eps = self.data.num_episodes()
        if not self.stream:
            if self.episode_done:
                if self.random:
                    # select random episode
                    self.episode_idx = random.randrange(num_eps)
                else:
                    # select next episode
                    self.episode_idx = (self.episode_idx + self.step_size) % num_eps
                self.entry_idx = 0
            else:
                self.entry_idx += 1
            action, epoch_done = self.data.get(self.episode_idx, self.entry_idx)
        else:
            action, epoch_done = self.data.get()

        if self.random:
            epoch_done = False
        elif (self.episode_idx + self.step_size >= num_eps and
                action['episode_done'] and not self.stream):
            # this is used for ordered data to check whether there's more data
            epoch_done = True

        return action, epoch_done

    def act(self):
        """Send new dialog message."""
        if self.epochDone:
            return {'episode_done': True}
        action, self.epochDone = self.next_example()
        self.episode_done = action['episode_done']
        action['id'] = self.getID()
        self.lastY = action.get('labels', None)
        if not self.datatype.startswith('train'):
            action.pop('labels', None)
        return action

    # Return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        return self.metrics.report()


class DialogData(object):
    """Provides a data structure for accessing textual dialog data.
    This can be used whenever the dialog data is a fixed log of chats
    (i.e not a simulator setting). The logs can include dialog text and possibly
    supervised labels, candidate labels and rewards.

    All these are stored in this internal data format which is used by the
    ``DialogTeacher`` class.

    ``data_loader`` is an iterable, with each call returning:

        ``(x, ...), new_episode?``

        Where

        - ``x`` is a query and possibly context

        ``...`` can contain additional fields, specifically

        - ``y`` is an iterable of label(s) for that query
        - ``r`` is the str reward for getting that query correct
        - ``c`` is an iterable of label candidates that the student can choose from
        - ``i`` is a str path to an image on disk, which will be loaded by the data
          class at request-time. should always point to the raw image file.
        - ``new_episode?`` is a boolean value specifying whether that example is the start of a new episode. If you don't use episodes set this to ``True`` every time.


    ``cands`` can be set to provide a list of candidate labels for every example
    in this dataset, which the agent can choose from (the correct answer
    should be in this set).


    ``random`` tells the data class whether or not to visit episodes sequentially
    or randomly when returning examples to the caller.
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
            self.cands = None if cands == None else set(sys.intern(c) for c in cands)
        self.addedCands = []
        self.copied_cands = False

    def share(self):
        shared = {
            'data': self.data,
            'cands': self.cands,
            'image_loader': self.image_loader
        }
        return shared

    def __len__(self):
        """Returns total number of entries available. Each episode has at least
        one entry, but might have many more.
        """
        return sum(len(episode) for episode in self.data)

    def _read_episode(self, data_generator):
        """Reads one episode at a time from the provided iterator over entries.
        """
        episode = []
        last_cands = None
        for entry, new in data_generator:
            if new and len(episode) > 0:
                yield tuple(episode)
                episode = []
                last_cands = None

            # intern all strings so we don't store them more than once
            new_entry = []
            if len(entry) > 0:
                # process text if available
                if entry[0] is not None:
                    new_entry.append(sys.intern(entry[0]))
                else:
                    new_entry.append(None)
                if len(entry) > 1:
                    # process labels if available
                    if entry[1] is None:
                        new_entry.append(None)
                    elif hasattr(entry[1], '__iter__') and type(entry[1]) is not str:
                        # make sure iterable over labels, not single string
                        new_entry.append(tuple(sys.intern(e) for e in entry[1]))
                    else:
                        raise TypeError('Must provide iterable over labels, not a single string.')
                    if len(entry) > 2:
                        # process reward if available
                        if entry[2] is not None:
                            new_entry.append(sys.intern(entry[2]))
                        else:
                            new_entry.append(None)
                        if len(entry) > 3:
                            # process label candidates if available
                            if entry[3] is None:
                                new_entry.append(None)
                            elif last_cands and entry[3] is last_cands:
                                # if cands are shared, say "same" so we
                                # don't store them again
                                new_entry.append(
                                    sys.intern('same as last time'))
                            elif hasattr(entry[3], '__iter__') and type(entry[3]) is not str:
                                # make sure iterable over candidates, not single string
                                last_cands = entry[3]
                                new_entry.append(tuple(
                                    sys.intern(e) for e in entry[3]))
                            else:
                                raise TypeError('Must provide iterable over label candidates, not a single string.')
                            if len(entry) > 4 and entry[4] is not None:
                                new_entry.append(sys.intern(entry[4]))

            episode.append(tuple(new_entry))

        if len(episode) > 0:
            yield tuple(episode)

    def _load(self, data_loader, datafile):
        """Loads up data from an iterator over tuples described in the class
        docs.
        """
        for episode in self._read_episode(data_loader(datafile)):
            self.data.append(episode)

    def num_episodes(self):
        """Return number of episodes in the dataset."""
        return len(self.data)

    def get(self, episode_idx, entry_idx=0):
        """Returns a specific entry from the dataset."""
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
        """Packs an entry into an action-observation dictionary."""
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

        if (table.get('labels', None) is not None
                and self.cands is not None):
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
    """Provides a data structure for streaming textual dialog data.
    This can be used whenever the dialog data follows the format described in
    DialogData but cannot fit entirely into memory.

    Additional keyword-argument cycle defines if the stream should restart from
    the beginning after an epoch is finished (defaults to False).
    """

    def __init__(self, opt, data_loader=None, cands=None, shared=None, **kwargs):
        self.cycle = kwargs['cycle'] if 'cycle' in kwargs else False
        self.data_loader = data_loader
        if shared:
            # auxiliary instances hold pointer to main datastream (in self.data)
            self.reset_data = shared['reset']
            self.file_queue = shared['file_queue']
            self.num_file_queue_left = shared['num_file_queue_left']
        else:
            # main instance holds the stream and shares pointer to it
            self.datafile = opt['datafile']
            self.file_queue = Queue()
            self.num_file_queue_left = Value('i', 0)
            file_list = []
            if os.path.isfile(self.datafile):
                file_list.append((self.datafile, os.stat(self.datafile).st_size))
            elif os.path.isdir(self.datafile):
                for dirpath, _, filenames in os.walk(self.datafile):
                    for f in filenames:
                        full_path = os.path.join(dirpath, f)
                        file_list.append((full_path, os.stat(full_path).st_size))
            else:
                raise RuntimeError("input file does not exist.")
            with self.num_file_queue_left.get_lock():
                self.num_file_queue_left.value += len(file_list)
            file_list.sort(key=lambda x: -x[1])
            for _filename, _size in file_list:
                self.file_queue.put(_filename)
            self.reset_data = None
            self.is_reset = True
        # super() call initiates stream in self.data by calling _load()
        super().__init__(opt, data_loader, cands, shared, **kwargs)
        self.entry_idx = 0
        self.next_episode = None
        self.cur_file = None

    def share(self):
        shared = super().share()
        # put back the file
        if self.cur_file:
            self.file_queue.put(self.cur_file)
            self.cur_file = None
            with self.num_file_queue_left.get_lock():
                self.num_file_queue_left += 1
        shared['file_queue'] = self.file_queue
        shared['num_file_queue_left'] = self.num_file_queue_left
        # also share reset method to allow datastream to be reset
        shared['reset'] = self.reset
        return shared

    def __len__(self):
        # unknown
        return 0

    def _load(self, data_loader, datafile):
        """Load data generator into data field."""
        self.data = self._data_generator(data_loader, datafile)

    def _data_generator(self, data_loader, datafile):
        """Generates data using the iterator over tuples constructed
        by data_loader.
        """
        self.is_reset = False
        while True:
            with self.num_file_queue_left.get_lock():
                num_left = self.num_file_queue_left.value
                self.num_file_queue_left.value -= 1
            if num_left % 50 == 0:
                logging.info("%d files left..." % num_left)
            if num_left <= 0:
                while True:
                    yield ((None,),)
            else:
                datafile = self.file_queue.get()
                self.cur_file = datafile
                for episode in self._read_episode(data_loader(datafile)):
                    yield episode

    def num_episodes(self):
        # unknown
        return 0

    def get(self):
        """Returns a the next entry from the stream in the current episode for
        this instance. When episode is done returns first entry of next episode.
        """
        # first look up data
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
        end_of_data = episode_done and self.next_episode[0][0] is None
        if end_of_data and self.cycle:
            self.next_episode = next(self.data)

        # last entry in this episode
        table['episode_done'] = episode_done
        return table, end_of_data

    def reset(self):
        """Reset the datastream to its beginning"""
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

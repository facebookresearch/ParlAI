# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from .agents import Teacher
from .thread_utils import SharedTable
from .metrics import Metrics

import copy
import random
import sys
import time


class DialogTeacher(Teacher):
    """A base teacher class for doing dialog with fixed chat logs.
    This class provides a set a basic functionality:
    - uses data class to store and query text data
    - generates action tables to send to the student agent from the data
    - metrics tracking count of sent vs correctly answered queries

    If you have opt.numthreads > 1, this also activates a shared memory
    array for the data and lock-protected shared-memory metrics.

    In order to subclass this class, you must implement setup_data() in your
    class (or subclass another class which does, like FbDialogTeacher), which
    reads your data file as an iterator. See the data module for a description
    of the requirements for setup_data().
    """

    def __init__(self, opt, shared=None):
        # Check for setup_data
        print("[DialogTeacher initializing.]")
        if not hasattr(self, 'setup_data'):
            raise RuntimeError('Must implement setup_data or subclass a class' +
                               ' which implements it (e.g. FbDialogTeacher)' +
                               ' in order to use this class.')

        super().__init__(opt, shared)

        self.datatype = opt['datatype']
        self.startTime = time.time()

        # first initialize any shared objects
        self.random = self.datatype == 'train'
        if shared and shared.get('data'):
            self.data = shared['data']
        else:
            self.data = DialogData(self.setup_data(opt['datafile']),
                                   cands=self.label_candidates())

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
        self.epochDone = False
        self.episode_done = True

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.epochDone = False
        return self

    def __next__(self):
        if self.epochDone:
            raise StopIteration()

    # share datatype, data, metrics, and a lock on the metrics
    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared

    def label_candidates(self):
        """Returns None by default, but override this in children (such as
        FbDialogTeacher) to load up candidate labels for every example.
        """
        return None

    def observe(self, observation):
        """Process observation for metrics. """
        if self.lastY is not None:
            loss = self.metrics.update(observation, self.lastY)
            self.lastY = None

    def next_example(self):
        num_eps = self.data.num_episodes()
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

        if self.random:
            epoch_done = False
        elif (self.episode_idx + self.step_size >= num_eps and
                action['episode_done']):
            # this is used for ordered data to check whether there's more data
            epoch_done = True

        return action, epoch_done

    def act(self):
        """Send new dialog message."""
        if self.epochDone:
            return { 'episode_done': True }
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
    DialogTeacher class.

    data_loader is an iterable, with each call returning:

    (x, ...), new_episode?

    Where...
    x is a query and possibly context
    ... can contain additional fields, specifically
        y is an iterable of label(s) for that query
        r is the str reward for getting that query correct
        c is an iterable of label candidates that the student can choose from
    new_episode? is a boolean value specifying whether that example is the start
    of a new episode. If you don't use episodes set this to True every time.

    cands can be set to provide a list of candidate labels for every example
        in this dataset, which the agent can choose from (the correct answer
        should be in this set).

    random tells the data class whether or not to visit episodes sequentially
    or randomly when returning examples to the caller.
    """

    def __init__(self, data_loader, cands=None):
        self.data = []
        self._load(data_loader)
        self.cands = None if cands == None else set(sys.intern(c) for c in cands)
        self.addedCands = []

    def __len__(self):
        """Returns total number of entries available. Each episode has at least
        one entry, but might have many more.
        """
        length = 0
        for l in self.data:
            length += len(l)
        return length

    def _load(self, data_loader):
        """Loads up data from an iterator over tuples described in the class
        docs.
        """
        episode = []
        last_cands = None
        for entry, new in data_loader:
            if new:
                if len(episode) > 0:
                    self.data.append(tuple(episode))
                    episode = []
                    last_cands = None

            # intern all strings so we don't store them more than once
            new_entry = []
            if len(entry) > 0:
                # process text
                if entry[0] is not None:
                    new_entry.append(sys.intern(entry[0]))
                else:
                    new_entry.append(None)
                if len(entry) > 1:
                    # process labels
                    if entry[1] is not None:
                        new_entry.append(tuple(sys.intern(e) for e in entry[1]))
                    else:
                        new_entry.append(None)
                    if len(entry) > 2:
                        # process reward
                        if entry[2] is not None:
                            new_entry.append(sys.intern(entry[2]))
                        else:
                            new_entry.append(None)
                        if len(entry) > 3 and entry[3] is not None:
                            # process label candidates
                            if last_cands and entry[3] is last_cands:
                                new_entry.append(
                                    sys.intern('same as last time'))
                            else:
                                last_cands = entry[3]
                                new_entry.append(tuple(
                                    sys.intern(e) for e in entry[3]))
            episode.append(tuple(new_entry))

        if len(episode) > 0:
            self.data.append(tuple(episode))

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
        table = {}
        table['text'] = entry[0]
        if len(entry) > 1:
            table['labels'] = entry[1]
            if len(entry) > 2:
                table['reward'] = entry[2]
                if len(entry) > 3:
                    table['label_candidates'] = entry[3]

        if (table.get('labels', None) is not None
            and self.cands is not None):
            if self.addedCands:
                # remove elements in addedCands
                self.cands.difference_update(self.addedCands)
                self.addedCands.clear()
            for label in table['labels']:
                if label not in self.cands:
                    # add labels, queue them for removal next time
                    self.cands.add(label)
                    self.addedCands.append(label)
            table['label_candidates'] = self.cands

        if 'labels' in table and 'label_candidates' in table:
            if table['labels'][0] not in table['label_candidates']:
                raise RuntimeError('true label missing from candidate labels')

        # last entry in this episode
        table['episode_done'] = episode_done
        return table, end_of_data

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from .agents import Teacher
from .data import TextData
from .thread_utils import SharedTable
from .metrics import Metrics

import copy
import random
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
        self.opt = copy.deepcopy(opt)
        print("[DialogTeacher initializing.]")
        if not hasattr(self, 'setup_data'):
            raise RuntimeError('Must implement setup_data or subclass a class' +
                               ' which implements it (e.g. FbDialogTeacher)' +
                               ' in order to use this class.')

        self.datatype = opt['datatype']
        self.startTime = time.time()
        if not hasattr(self, 'id'):
            self.id = opt.get('task', 'teacher')

        # first initialize any shared objects
        self.random = self.datatype == 'train'
        if shared and shared.get('data'):
            self.data = shared['data']
        else:
            self.data = TextData(self.setup_data(opt['datafile']),
                                 cands=self.label_candidates())

        if shared and shared.get('metrics'):
            self.metrics = shared['metrics']
        else:
            self.metrics = Metrics(opt)

        self.reset()

    def reset(self):
        # Reset the dialog so that it is at the start of the epoch,
        # and all metrics are reset.
        self.metrics.clear()
        self.lastY = None
        self.episode_idx = -1
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
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        shared['data'] = self.data
        shared['metrics'] = self.metrics
        return shared

    def label_candidates(self):
        """Returns None by default, but override this in children (such as
        FbDialogTeacher) to load up candidate labels for every example.
        """
        return None

    def observe(self, observation):
        """Store observation and process for metrics. """
        self.observation = observation
        if self.lastY is not None:
            obs = self.observation if hasattr(self, 'observation') else {}
            loss = self.metrics.update(
                obs, self.lastY, self.lastLabelCandidates)
            self.lastY = None
            self.lastLabelCandidates = None

    def next_example(self):
        if self.episode_done:
            num_eps = self.data.num_episodes()
            if self.random:
                # select random episode
                self.episode_idx = random.randrange(num_eps)
            else:
                # select next episode
                self.episode_idx = (self.episode_idx + 1) % num_eps
            self.entry_idx = 0
        else:
            self.entry_idx += 1
        return self.data.get(self.episode_idx, self.entry_idx)

    def act(self):
        """Send new dialog message. """
        action, self.epochDone = self.next_example()
        self.episode_done = action['episode_done']
        action['id'] = self.getID()
        self.lastY = action.get('labels', None)
        self.lastLabelCandidates = action.get('label_candidates', None)
        if not self.datatype.startswith('train'):
            action.pop('labels', None)
        return action

    # Return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        return self.metrics.report()

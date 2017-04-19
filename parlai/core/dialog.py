#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import copy
import time

from .agents import Teacher
from .data import TextData, HogwildTextData
from .thread_utils import SharedTable
from .metrics import Metrics


class DialogTeacher(Teacher):
    """This class provides a set a basic functionality:
    - metrics tracking count of sent vs correctly answered queries
    - uses data class to store and query text data
    - generates action tables to send to the student agent from the data

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
        self.fin = False
        self.lastY = None
        self.id = opt['task']

        # first initialize any shared objects
        if shared and shared.get('data'):
            self.data = shared['data']
        else:
            if True or opt.get('numthreads', 1) == 1:
                self.data = TextData(self.setup_data(opt['datafile']),
                                     cands=self.label_candidates(),
                                     random=self.datatype == 'train')
            else:
                self.data = HogwildTextData(self.setup_data(opt['datafile']),
                                            cands=self.label_candidates(),
                                            random=self.datatype == 'train')

        if shared and shared.get('metrics'):
            self.metrics = shared['metrics']
        else:
            self.metrics = Metrics(opt)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.fin = False
        return self

    def __next__(self):
        if self.fin:
            raise StopIteration()

    # share datatype, data, metrics, and a lock on the metrics
    def share(self):
        shared = {}
        shared['class'] = type(self)
        shared['opt'] = self.opt
        if self.opt.get('numthreads', 1) > 1:
            shared['data'] = self.data
            shared['metrics'] = self.metrics
        return shared

    def label_candidates(self):
        """Returns None by default, but override this in children (such as
        FbDialogTeacher) to load up candidate labels for every example.
        """
        return None

    # Check received text for correct answer then send new query.
    def act(self):
        # First process observation for metrics.
        if self.lastY is not None:
            loss = self.metrics.update(
                self.observation, self.lastY, self.lastLabelCandidates)
            self.lastY = None
            self.lastLabelCandidates = None

        # Then build reply.
        action, self.fin = next(self.data)
        action['id'] = self.getID()
        self.lastY = action.get('labels', None)
        self.lastLabelCandidates = action.get('label_candidates', None)
        if not self.datatype.startswith('train'):
            action.pop('labels', None)
        return action

    # Return transformed metrics showing total examples and accuracy if avail.
    def report(self):
        report = self.metrics.report()
        return report

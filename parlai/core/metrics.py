# Copyright 2004-present Facebook. All Rights Reserved.
"""Provides standard metric evaluations for dialog.
Uses locking and shared memory when numthreads is set to >1 to share metrics
between processes.
"""

from .thread_utils import SharedTable
import copy
import importlib
import random
import re

_re_alphanumeric = re.compile('[^a-zA-Z0-9_]+')

def _check_answer(guess, answers):
    if guess is None or answers is None:
        return False
    # either validating or testing--check correct answers
    r_test = _re_alphanumeric.sub('', guess.lower())
    for answer in answers:
        if _re_alphanumeric.sub('', answer.lower()) == r_test:
            return True
    return False


class Metrics(object):
    """Class that maintains evaluation metrics over dialog."""

    def __init__(self, opt):
        self.metrics = {}
        self.metrics['cnt'] = 0
        self.metrics['correct'] = 0
        if opt.get('numthreads', 1) > 1:
            self.metrics = SharedTable(self.metrics)
        self.datatype = opt.get('datatype', 'train')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def _lock(self):
        if hasattr(self.metrics, 'get_lock'):
            # use the shared_table's lock
            return self.metrics.get_lock()
        else:
            # otherwise do nothing
            return self

    def update(self, prediction=None, labels=None):
        with self._lock():
            self.metrics['cnt'] += 1
        correct = 0
        if _check_answer(prediction, labels):
            correct = 1
        with self._lock():
            self.metrics['correct'] += correct
        # Return a dict containing the metrics for this specific example.
        # Metrics across all data is stored internally in the class, and
        # can be accessed with the report method.
        loss = {}
        loss['correct'] = correct
        return loss

    def report(self):
        # Report the metrics over all data seen so far.
        m = {}
        m['total'] = self.metrics['cnt']
        if self.metrics['cnt'] > 0:
            m['accuracy'] = self.metrics['correct'] / self.metrics['cnt']
        return m

    def clear(self):
        with self._lock():
            self.metrics['cnt'] = 0
            self.metrics['correct'] = 0

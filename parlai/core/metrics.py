# Copyright 2004-present Facebook. All Rights Reserved.
"""Provides standard metric evaluations for dialog.
"""

import copy
import importlib
import random
import re

_re_alphanumeric = re.compile('[^a-zA-Z0-9_]+')

def _check_answer(guess, answers):
    # either validating or testing--check correct answers
    r_test = _re_alphanumeric.sub('', guess.lower())
    for answer in answers:
        if _re_alphanumeric.sub('', answer.lower()) == r_test:
            return True
    return False


class Metrics(object):
    """Class that maintains evaluation metrics over dialog."""

    def __init__(self, opt):
        print("[Metrics initializing.]")
        self.metrics = {}
        self.metrics['cnt'] = 0
        self.metrics['correct'] = 0
        self.datatype = opt['datatype']
        pass

    def update(self, prediction, labels):
        self.metrics['cnt'] += 1
        correct = 0
        if _check_answer(prediction, labels):
            correct = 1
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
        if self.metrics['correct'] > 0 or not self.datatype.startswith('train'):
            m['accuracy'] = self.metrics['correct'] / self.metrics['cnt']
        return m

    def clear(self):
        self.metrics['cnt'] = 0
        self.metrics['correct'] = 0


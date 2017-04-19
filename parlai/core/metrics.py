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
        self.eval_pr = [1, 5, 10, 50, 100]
        for k in self.eval_pr:
            self.metrics['hits@' + str(k)] = 0
        if opt.get('numthreads', 1) > 1:
            self.metrics = SharedTable(self.metrics)
        self.datatype = opt.get('datatype', 'train')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def __str__(self):
        return str(self.metrics)

    def __repr__(self):
        return repr(self.metrics)

    def _lock(self):
        if hasattr(self.metrics, 'get_lock'):
            # use the shared_table's lock
            return self.metrics.get_lock()
        else:
            # otherwise do nothing
            return self

    def update_ranking_metrics(self, observation, labels, label_cands):
        text_cands = observation.get('text_candidates', None)
        if text_cands is None:
            text = observation.get('text', None)
            if text is None:
                return
            else:
                text_cands = [ text ]
        # Now loop through text candidates, assuming they are sorted.
        # If any of them is a label then score a point.
        # maintain hits@1, 5, 10, 50, 100,  etc.
        label_set = set(labels) if type(labels) != set else labels
        cnts = {k: 0 for k in self.eval_pr}
        cnt = 0
        for c in text_cands:
            cnt += 1
            if c in label_set:
                for k in self.eval_pr:
                    if cnt <= k:
                        cnts[k] += 1
        # hits metric is 1 if cnts[k] > 0.
        # (other metrics such as p@k and r@k take
        # the value of cnt into account.) 
        with self._lock():
            for k in self.eval_pr:
                if cnts[k] > 0:
                    self.metrics['hits@' + str(k)] += 1


    def update(self, observation, labels, label_cands):
        with self._lock():
            self.metrics['cnt'] += 1

        # Exact match metric.
        correct = 0
        prediction = observation.get('text', None)
        if _check_answer(prediction, labels):
            correct = 1
        with self._lock():
            self.metrics['correct'] += correct

        # Ranking metrics.
        self.update_ranking_metrics(observation, labels, label_cands)

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
            m['hits@k'] = {}
            for k in self.eval_pr:
                m['hits@k'][k] = self.metrics['hits@' + str(k)] / self.metrics['cnt']
        return m

    def clear(self):
        with self._lock():
            self.metrics['cnt'] = 0
            for k in self.eval_pr:
                self.metrics['hits@' + str(k)][k] = 0
            self.metrics['correct'] = 0

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Provides standard metric evaluations for dialog.
Uses locking and shared memory when ``numthreads`` is set to >1 to share metrics
between processes.
"""

from parlai.core.thread_utils import SharedTable
from parlai.core.utils import round_sigfigs
from collections import Counter

import re

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
def _normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(guess, answers):
    """Check if guess is a (normalized) exact match with any answer."""
    if guess is None or answers is None:
        return False
    guess = _normalize_answer(guess)
    for a in answers:
        if guess == _normalize_answer(a):
            return True
    return False


def _f1_score(guess, answers):
    """Return the max F1 score between the guess and any answer."""
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(g_tokens)
        recall = 1.0 * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if guess is None or answers is None:
        return 0
    g_tokens = _normalize_answer(guess).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


class Metrics(object):
    """Class that maintains evaluation metrics over dialog."""

    def __init__(self, opt):
        self.metrics = {}
        self.metrics['cnt'] = 0
        self.metrics['correct'] = 0
        self.metrics['f1'] = 0.0
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

    def update_ranking_metrics(self, observation, labels):
        text_cands = observation.get('text_candidates', None)
        if text_cands is None:
            text = observation.get('text', None)
            if text is None:
                return
            else:
                text_cands = [text]
        # Now loop through text candidates, assuming they are sorted.
        # If any of them is a label then score a point.
        # maintain hits@1, 5, 10, 50, 100,  etc.
        label_set = set(_normalize_answer(l) for l in labels)
        cnts = {k: 0 for k in self.eval_pr}
        cnt = 0
        for c in text_cands:
            cnt += 1
            if _normalize_answer(c) in label_set:
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

    def update(self, observation, labels):
        with self._lock():
            self.metrics['cnt'] += 1

        # Exact match metric.
        correct = 0
        prediction = observation.get('text', None)
        if _exact_match(prediction, labels):
            correct = 1
        with self._lock():
            self.metrics['correct'] += correct

        # F1 metric.
        f1 = _f1_score(prediction, labels)
        with self._lock():
            self.metrics['f1'] += f1

        # Ranking metrics.
        self.update_ranking_metrics(observation, labels)

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
            m['accuracy'] = round_sigfigs(
                self.metrics['correct'] / self.metrics['cnt'], 4)
            m['f1'] = round_sigfigs(
                self.metrics['f1'] / self.metrics['cnt'], 4)
            m['hits@k'] = {}
            for k in self.eval_pr:
                m['hits@k'][k] = round_sigfigs(
                    self.metrics['hits@' + str(k)] / self.metrics['cnt'], 4)
        return m

    def clear(self):
        with self._lock():
            self.metrics['cnt'] = 0
            self.metrics['correct'] = 0
            self.metrics['f1'] = 0.0
            for k in self.eval_pr:
                self.metrics['hits@' + str(k)] = 0

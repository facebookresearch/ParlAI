#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Provides standard metric evaluations for dialog.

Uses locking and shared memory when ``numthreads`` is set to >1 to share metrics
between processes.
"""

from parlai.core.thread_utils import SharedTable
from parlai.core.utils import round_sigfigs, no_lock
from collections import Counter
from parlai.core.utils import warn_once

import re

try:
    from nltk.translate import bleu_score as nltkbleu
except ImportError:
    # User doesn't have nltk installed, so we can't use it for bleu
    # We'll just turn off things, but we might want to warn the user
    nltkbleu = None

try:
    import rouge as rouge
except ImportError:
    # User doesn't have rouge installed, so we can't use it for rouge
    # We'll just turn off things, but we might want to warn the user
    warn_once('Rouge metrics require py-rouge. Please run `pip install py-rouge`.')
    rouge = None

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
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


def aggregate_task_reports(reports, tasks, micro=True):
    """
    Aggregate separate task reports into a single report.

    :param reports: list of report dicts from separate tasks
    :param tasks: list of tasks
    :param micro: average per example if True, else average over t

    :return: aggregated report dicts
    """
    if len(reports) == 1:
        # singular task
        return reports[0]
    # multiple tasks, aggregate metrics
    metrics = {}
    exs = {}
    total_report = {'tasks': {}}
    # collect metrics from all reports
    for i, report in enumerate(reports):
        total_report['tasks'][tasks[i]] = report
        for metric, val in report.items():
            if metric == 'exs':
                exs[tasks[i]] = val
            else:
                metrics.setdefault(metric, {})[tasks[i]] = val
    # now aggregate
    total_exs = sum(exs.values())
    total_report['exs'] = total_exs
    for metric, task_vals in metrics.items():
        if micro:
            # average over the number of examples
            vals = [task_vals[task] * exs[task] for task in tasks]
            total_report[metric] = round_sigfigs(sum(vals) / total_exs, 4)
        else:  # macro
            # average over tasks
            vals = task_vals.values()
            total_report[metric] = round_sigfigs(sum(vals) / len(vals), 4)
    # add a warning describing how metrics were averaged across tasks.
    total_report['warning'] = 'metrics are averaged across tasks'
    if micro:
        total_report['warning'] += (' and weighted by the number of examples '
                                    'per task')
    return total_report


def _exact_match(guess, answers):
    """Check if guess is a (normalized) exact match with any answer."""
    if guess is None or answers is None:
        return False
    guess = normalize_answer(guess)
    for a in answers:
        if guess == normalize_answer(a):
            return True
    return False


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split())for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


def _bleu(guess, answers):
    """Compute approximate BLEU score between guess and a set of answers."""
    if nltkbleu is None:
        # bleu library not installed, just return a default value
        return None
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    return nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
    )


def _rouge(guess, answers):
    global rouge
    """Compute ROUGE score between guess and *any* answers. Return the best."""
    if rouge is None:
        return None, None, None
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    try:
        scores = [evaluator.get_scores(normalize_answer(guess), normalize_answer(a))
                  for a in answers]
    except LookupError:
        warn_once(
            'ROUGE requires nltk punkt tokenizer. Please run '
            '`python -c "import nltk; nltk.download(\'punkt\')`'
        )
        rouge = None
        return None, None, None

    scores_rouge1 = [score['rouge-1']['r'] for score in scores]
    scores_rouge2 = [score['rouge-2']['r'] for score in scores]
    scores_rougel = [score['rouge-l']['r'] for score in scores]
    return max(scores_rouge1), max(scores_rouge2), max(scores_rougel)


def aggregate_metrics(reporters):
    """Aggregate metrics from multiple reports."""
    # reporters is a list of teachers or worlds
    m = {}
    m['tasks'] = {}
    sums = {'accuracy': 0, 'f1': 0, 'loss': 0, 'ppl': 0}
    if nltkbleu is not None:
        sums['bleu'] = 0
    if rouge is not None:
        sums['rouge-1'] = 0.0
        sums['rouge-2'] = 0.0
        sums['rouge-L'] = 0.0
    num_tasks = 0
    total = 0
    for i in range(len(reporters)):
        task_id = reporters[i].getID()
        task_report = reporters[i].report()
        while task_id in m['tasks']:
            # prevent name clobbering if using multiple tasks with same ID
            task_id += '_'
        m['tasks'][task_id] = task_report
        total += task_report['exs']
        found_any = False
        for k in sums.keys():
            if k in task_report:
                sums[k] += task_report[k]
                found_any = True
        if found_any:
            num_tasks += 1
    m['exs'] = total
    m['accuracy'] = 0
    if num_tasks > 0:
        for k in sums.keys():
            m[k] = round_sigfigs(sums[k] / num_tasks, 4)
    return m


class Metrics(object):
    """Class that maintains evaluation metrics over dialog."""

    def __init__(self, opt):
        self.metrics = {}
        self.metrics['cnt'] = 0
        self.metrics_list = ['mean_rank', 'loss', 'correct', 'f1', 'ppl']
        if nltkbleu is not None:
            # only compute bleu if we can
            self.metrics_list.append('bleu')
        if rouge is not None:
            # only compute rouge if we can
            self.metrics_list.append('rouge-1')
            self.metrics_list.append('rouge-2')
            self.metrics_list.append('rouge-L')
        for k in self.metrics_list:
            self.metrics[k] = 0.0
            self.metrics[k + '_cnt'] = 0
        self.eval_pr = [1, 5, 10, 100]
        for k in self.eval_pr:
            self.metrics['hits@' + str(k)] = 0
        self.metrics['hits@_cnt'] = 0
        self.flags = {'has_text_cands': False, 'print_prediction_metrics': False}
        if opt.get('numthreads', 1) > 1:
            self.metrics = SharedTable(self.metrics)
            self.flags = SharedTable(self.flags)

    def __str__(self):
        return str(self.metrics)

    def __repr__(self):
        representation = super().__repr__()
        return representation.replace('>', ': {}>'.format(repr(self.metrics)))

    def _lock(self):
        if hasattr(self.metrics, 'get_lock'):
            # use the shared_table's lock
            return self.metrics.get_lock()
        else:
            # otherwise do nothing
            return no_lock()

    def _update_ranking_metrics(self, observation, labels):
        text_cands = observation.get('text_candidates', None)
        if text_cands is None:
            return
        else:
            # Now loop through text candidates, assuming they are sorted.
            # If any of them is a label then score a point.
            # maintain hits@1, 5, 10, 50, 100,  etc.
            label_set = set(normalize_answer(l) for l in labels)
            cnts = {k: 0 for k in self.eval_pr}
            cnt = 0
            for c in text_cands:
                cnt += 1
                if normalize_answer(c) in label_set:
                    for k in self.eval_pr:
                        if cnt <= k:
                            cnts[k] += 1
            # hits metric is 1 if cnts[k] > 0.
            # (other metrics such as p@k and r@k take
            # the value of cnt into account.)
            with self._lock():
                self.flags['has_text_cands'] = True
                for k in self.eval_pr:
                    if cnts[k] > 0:
                        self.metrics['hits@' + str(k)] += 1
                self.metrics['hits@_cnt'] += 1

    def update(self, observation, labels):
        """Update metrics based on an observation and true labels."""
        with self._lock():
            self.metrics['cnt'] += 1

        # Exact match metric.
        correct = 0
        prediction = observation.get('text', None)
        if prediction is not None:
            if _exact_match(prediction, labels):
                correct = 1
            with self._lock():
                self.flags['print_prediction_metrics'] = True
                self.metrics['correct'] += correct
                self.metrics['correct_cnt'] += 1

            # F1 and BLEU metrics.
            f1 = _f1_score(prediction, labels)
            bleu = _bleu(prediction, labels)
            rouge1, rouge2, rougel = _rouge(prediction, labels)

            with self._lock():
                self.metrics['f1'] += f1
                self.metrics['f1_cnt'] += 1
                if bleu is not None:
                    self.metrics['bleu'] += bleu
                    self.metrics['bleu_cnt'] += 1
                if rouge1 is not None:
                    self.metrics['rouge-1'] += rouge1
                    self.metrics['rouge-2'] += rouge2
                    self.metrics['rouge-L'] += rougel
                    self.metrics['rouge-1_cnt'] += 1
                    self.metrics['rouge-2_cnt'] += 1
                    self.metrics['rouge-L_cnt'] += 1

        # Ranking metrics.
        self._update_ranking_metrics(observation, labels)

        # User-reported metrics
        if 'metrics' in observation:
            for k, v in observation['metrics'].items():
                if k not in ['correct', 'f1', 'hits@k', 'bleu', 'rouge-1',
                             'rouge-2', 'rouge-L']:
                    if k in self.metrics_list:
                        with self._lock():
                            self.metrics[k] += v
                            self.metrics[k + '_cnt'] += 1
                    else:
                        if type(self.metrics) is SharedTable:
                            # can't share custom metrics during hogwild
                            pass
                        else:
                            # no need to lock because not SharedTable
                            if k not in self.metrics:
                                self.metrics[k] = v
                                self.metrics_list.append(k)
                                self.metrics[k + '_cnt'] = 1.0
                            else:
                                self.metrics[k] += v

        # Return a dict containing the metrics for this specific example.
        # Metrics across all data is stored internally in the class, and
        # can be accessed with the report method.
        loss = {}
        loss['correct'] = correct
        return loss

    def report(self):
        """Report the metrics over all data seen so far."""
        m = {}
        total = self.metrics['cnt']
        m['exs'] = total
        if total > 0:
            if self.flags['print_prediction_metrics']:
                m['accuracy'] = round_sigfigs(
                    self.metrics['correct'] / max(1, self.metrics['correct_cnt']),
                    4
                )
                m['f1'] = round_sigfigs(
                    self.metrics['f1'] / max(1, self.metrics['f1_cnt']),
                    4
                )
            if self.flags['has_text_cands']:
                for k in self.eval_pr:
                    m['hits@' + str(k)] = round_sigfigs(
                        self.metrics['hits@' + str(k)] /
                        max(1, self.metrics['hits@_cnt']),
                        3
                    )
            for k in self.metrics_list:
                if self.metrics[k + '_cnt'] > 0 and k != 'correct' and k != 'f1':
                    m[k] = round_sigfigs(
                        self.metrics[k] / max(1, self.metrics[k + '_cnt']),
                        4
                    )
        return m

    def clear(self):
        """Clear all the metrics."""
        # TODO: rename to reset for consistency with rest of ParlAI
        with self._lock():
            self.metrics['cnt'] = 0
            for k in self.metrics_list:
                v = self.metrics[k]
                v_typ = type(v)
                if 'Tensor' in str(v_typ):
                    self.metrics[k].zero_()
                else:
                    self.metrics[k] = 0.0
                self.metrics[k + '_cnt'] = 0
            for k in self.eval_pr:
                self.metrics['hits@' + str(k)] = 0
            self.metrics['hits@_cnt'] = 0

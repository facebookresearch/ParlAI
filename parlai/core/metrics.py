#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Provides standard metric evaluations for dialog.

Uses locking and shared memory when ``numthreads`` is set to >1 to share metrics between
processes.
"""

from parlai.utils.thread import SharedTable
from parlai.utils.misc import round_sigfigs, no_lock
from collections import Counter
from parlai.utils.misc import warn_once
from numbers import Number

import re

DEFAULT_METRICS = {'correct', 'bleu-4', 'accuracy', 'f1'}
ROUGE_METRICS = {'rouge-1', 'rouge-2', 'rouge-L'}
BLEU_METRICS = {'bleu-1', 'bleu-2', 'bleu-3'}
ALL_METRICS = DEFAULT_METRICS | ROUGE_METRICS | BLEU_METRICS


try:
    from nltk.translate import bleu_score as nltkbleu
except ImportError:
    # User doesn't have nltk installed, so we can't use it for bleu
    # We'll just turn off things, but we might want to warn the user
    nltkbleu = None

try:
    import rouge
except ImportError:
    # User doesn't have py-rouge installed, so we can't use it.
    # We'll just turn off rouge computations
    rouge = None

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def aggregate_task_reports(reports, tasks, micro=False):
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
        if all([isinstance(v, Number) for v in task_vals.values()]):
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
        total_report['warning'] += ' and weighted by the number of examples ' 'per task'
    return total_report


def _exact_match(guess, answers):
    """
    Check if guess is a (normalized) exact match with any answer.
    """
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
    """
    Return the max F1 score between the guess and *any* answer.
    """
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)


def _bleu(guess, answers, weights=None):
    """
    Compute approximate BLEU score between guess and a set of answers.
    """
    if nltkbleu is None:
        # bleu library not installed, just return a default value
        return None
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    if weights is None:
        # default bleu-4
        weights = [1 / 4 for _ in range(4)]
    return nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
        weights=weights,
    )


def _rouge(guess, answers):
    global rouge
    """Compute ROUGE score between guess and *any* answers. Return the best."""
    if rouge is None:
        return None, None, None
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    try:
        scores = [
            evaluator.get_scores(normalize_answer(guess), normalize_answer(a))
            for a in answers
        ]
    except LookupError:
        warn_once(
            'ROUGE requires nltk punkt tokenizer. Please run '
            '`python -c "import nltk; nltk.download(\'punkt\')`'
        )
        rouge = None
        return None, None, None

    scores_rouge1 = [score['rouge-1']['r'] for score in scores]
    scores_rouge2 = [score['rouge-2']['r'] for score in scores]
    scores_rougeL = [score['rouge-l']['r'] for score in scores]
    return max(scores_rouge1), max(scores_rouge2), max(scores_rougeL)


def aggregate_metrics(reporters):
    """
    Aggregate metrics from multiple reports.
    """
    # reporters is a list of teachers or worlds
    m = {}
    m['tasks'] = {}
    sums = {}
    num_tasks = 0
    total = 0
    for i in range(len(reporters)):
        task_id = reporters[i].getID()
        task_report = reporters[i].report()
        for each_metric, value in task_report.items():
            if isinstance(value, float):
                sums[each_metric] = 0.0
                m[each_metric] = 0.0
            elif isinstance(value, Number):
                sums[each_metric] = 0
                m[each_metric] = 0

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
    """
    Class that maintains evaluation metrics over dialog.
    """

    def __init__(self, opt):
        self.metrics = {}
        self.metrics['cnt'] = 0
        self.metrics_list = set()
        optional_metrics_list = []
        metrics_arg = opt.get('metrics', 'default')
        if metrics_arg == 'default':
            optional_metrics_list = DEFAULT_METRICS
        elif metrics_arg == 'all':
            optional_metrics_list = ALL_METRICS
        else:
            optional_metrics_list = set(metrics_arg.split(','))
            optional_metrics_list.add('correct')
        for each_m in optional_metrics_list:
            if each_m.startswith('rouge'):
                if rouge is not None:
                    # only compute rouge if rouge is available
                    self.metrics_list.add(each_m)
            elif each_m == 'bleu' and nltkbleu is None:
                # only compute bleu if bleu is available
                pass
            else:
                self.metrics_list.add(each_m)
        self._print_metrics_list = (
            self.metrics_list
            if 'rouge' not in self.metrics_list
            else self.metrics_list | ROUGE_METRICS
        )
        for k in self._print_metrics_list:
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
        """
        Update metrics based on an observation and true labels.
        """
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
            if 'f1' in self.metrics_list:
                f1 = _f1_score(prediction, labels)
            bleu_scores = {}
            rouge1 = rouge2 = rougeL = None
            if 'bleu-4' in self.metrics_list:
                bleu_scores['bleu-4'] = _bleu(prediction, labels)
            if 'bleu-1' in self.metrics_list:
                for i in range(3):
                    weights = [1 / (i + 1) for _ in range(i + 1)]
                    bleu_scores[f'bleu-{i + 1}'] = _bleu(prediction, labels, weights)
            if 'rouge-L' in self.metrics_list:
                rouge1, rouge2, rougeL = _rouge(prediction, labels)
            with self._lock():
                if 'f1' in self.metrics:
                    self.metrics['f1'] += f1
                    self.metrics['f1_cnt'] += 1
                if 'bleu-4' in self.metrics:
                    self.metrics['bleu-4'] += bleu_scores.pop('bleu-4')
                    self.metrics['bleu-4_cnt'] += 1
                if 'bleu-1' in self.metrics:
                    for b, b_score in bleu_scores.items():
                        self.metrics[b] += b_score
                        self.metrics[f'{b}_cnt'] += 1
                if 'rouge-L' in self.metrics and rouge1 is not None:
                    self.metrics['rouge-1'] += rouge1
                    self.metrics['rouge-1_cnt'] += 1
                    self.metrics['rouge-2'] += rouge2
                    self.metrics['rouge-2_cnt'] += 1
                    self.metrics['rouge-L'] += rougeL
                    self.metrics['rouge-L_cnt'] += 1

        # Ranking metrics.
        self._update_ranking_metrics(observation, labels)

        # User-reported metrics
        if 'metrics' in observation:
            for k, v in observation['metrics'].items():
                if k not in ALL_METRICS and k != 'rouge':
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
                                self.metrics_list.add(k)
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
        """
        Report the metrics over all data seen so far.
        """
        m = {}
        total = self.metrics['cnt']
        m['exs'] = total
        if total > 0:
            if self.flags['print_prediction_metrics']:
                if 'accuracy' in self.metrics_list:
                    m['accuracy'] = round_sigfigs(
                        self.metrics['correct'] / max(1, self.metrics['correct_cnt']), 4
                    )
                if 'f1' in self.metrics_list:
                    m['f1'] = round_sigfigs(
                        self.metrics['f1'] / max(1, self.metrics['f1_cnt']), 4
                    )
            if self.flags['has_text_cands']:
                for k in self.eval_pr:
                    m['hits@' + str(k)] = round_sigfigs(
                        self.metrics['hits@' + str(k)]
                        / max(1, self.metrics['hits@_cnt']),
                        3,
                    )
            for k in self._print_metrics_list:
                if self.metrics[k + '_cnt'] > 0 and k != 'correct' and k != 'f1':
                    m[k] = round_sigfigs(
                        self.metrics[k] / max(1, self.metrics[k + '_cnt']), 4
                    )
        return m

    def clear(self):
        """
        Clear all the metrics.
        """
        # TODO: rename to reset for consistency with rest of ParlAI
        with self._lock():
            self.metrics['cnt'] = 0
            metrics_list = (
                self.metrics_list
                if 'rouge' not in self.metrics_list
                else self.metrics_list | ROUGE_METRICS
            )
            for k in metrics_list:
                v = self.metrics[k]
                v_typ = type(v)
                if 'Tensor' in str(v_typ):
                    self.metrics[k].zero_()
                if isinstance(v, int):
                    self.metrics[k] = 0
                else:
                    self.metrics[k] = 0.0
                self.metrics[k + '_cnt'] = 0
            for k in self.eval_pr:
                self.metrics['hits@' + str(k)] = 0
            self.metrics['hits@_cnt'] = 0

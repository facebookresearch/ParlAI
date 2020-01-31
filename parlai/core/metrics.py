#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Provides standard metric evaluations for dialog.

Uses locking and shared memory when ``numthreads`` is set to >1 to share metrics between
processes.
"""

import re
from abc import ABC, abstractmethod
from collections import Counter
from numbers import Number
import queue
import functools
from typing import Union, List, Optional, Tuple, Set, Any

import torch

from parlai.core.message import Message
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.utils.typing import TScalar

try:
    import torch.multiprocessing as multiprocessing
except ImportError:
    import multiprocessing  # type: ignore


DEFAULT_METRICS = {'bleu-4', 'accuracy', 'f1'}
ROUGE_METRICS = {'rouge-1', 'rouge-2', 'rouge-L'}
BLEU_METRICS = {'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'}
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


@functools.total_ordering
class Metric(ABC):
    """
    Base class for storing metrics.

    Subclasses should define .value().
    """

    @abstractmethod
    def value(self) -> float:
        """
        Return the value of the metric as a float.
        """
        pass

    def __iadd__(self, other):
        return self.__radd__(other)

    def __radd__(self, other):
        if other is None:
            return self
        return self.__add__(other)

    def __str__(self) -> str:
        return f'{self.value():.4g}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value():.4g})'

    def __float__(self) -> float:
        return self.value()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() == other.value()
        else:
            return self.value() == other

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Metric):
            return self.value() < other.value()
        else:
            return self.value() < other

    @classmethod
    def as_number(cls, obj: TScalar) -> Union[int, float]:
        if isinstance(obj, torch.Tensor):
            obj_as_number: Union[int, float] = obj.item()
        else:
            obj_as_number = obj  # type: ignore
        assert isinstance(obj_as_number, int) or isinstance(obj_as_number, float)
        return obj_as_number

    @classmethod
    def as_float(cls, obj: TScalar) -> float:
        return float(cls.as_number(obj))

    @classmethod
    def as_int(cls, obj: TScalar) -> int:
        return int(cls.as_number(obj))


class SumMetric(Metric):
    """
    Class that keeps a running sum of some metric.
    """

    __slots__ = ('_sum',)

    def __init__(self, sum_: TScalar = 0):
        if isinstance(sum_, torch.Tensor):
            self._sum = sum_.item()
        else:
            assert isinstance(sum_, (int, float))
            self._sum = sum_

    def __add__(self, other: Optional['SumMetric']) -> 'SumMetric':
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_sum = self._sum + other._sum
        return SumMetric(sum_=full_sum)

    def value(self) -> float:
        return self._sum


class AverageMetric(Metric):
    """
    Class that keeps a running average of some metric.
    """

    __slots__ = ('_numer', '_denom')

    def __init__(self, numer: TScalar, denom: TScalar = 1):
        self._numer = self.as_number(numer)
        self._denom = self.as_number(denom)

    def __add__(self, other: Optional['AverageMetric']) -> 'AverageMetric':
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        full_numer: TScalar = self._numer + other._numer
        full_denom: TScalar = self._denom + other._denom
        return AverageMetric(numer=full_numer, denom=full_denom)

    def value(self) -> float:
        return self._numer / self._denom


class F1Metric(AverageMetric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
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

    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'F1Metric':
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        g_tokens = normalize_answer(guess).split()
        scores = [
            F1Metric._prec_recall_f1_score(g_tokens, normalize_answer(a).split())
            for a in answers
        ]
        return F1Metric(max(f1 for p, r, f1 in scores), 1)


class ExactMatchMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'ExactMatchMetric':
        if guess is None or answers is None:
            return None
        guess = normalize_answer(guess)
        for a in answers:
            if guess == normalize_answer(a):
                return ExactMatchMetric(1)
        return ExactMatchMetric(0)


class BleuMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str], k: int = 4) -> Optional['BleuMetric']:
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
        weights = [1 / k for _ in range(k)]
        score = nltkbleu.sentence_bleu(
            [normalize_answer(a).split(" ") for a in answers],
            normalize_answer(guess).split(" "),
            smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
            weights=weights,
        )
        return BleuMetric(score)


class RougeMetric(AverageMetric):
    _evaluator = None

    @staticmethod
    def compute_many(
        guess: str, answers: List[str]
    ) -> Tuple[
        Optional['RougeMetric'], Optional['RougeMetric'], Optional['RougeMetric']
    ]:
        """
        Compute ROUGE score between guess and *any* answer.

        Done with compute_many due to increased efficiency.

        :return: (rouge-1, rouge-2, rouge-L)
        """
        # possible global initialization
        global rouge
        if rouge is None:
            return None, None, None
        if RougeMetric._evaluator is None:
            RougeMetric._evaluator = rouge.Rouge(
                metrics=['rouge-n', 'rouge-l'], max_n=2
            )
        try:
            scores = [
                RougeMetric._evaluator.get_scores(
                    normalize_answer(guess), normalize_answer(a)
                )
                for a in answers
            ]
        except LookupError:
            warn_once(
                'ROUGE requires nltk punkt tokenizer. Please run '
                '`python -c "import nltk; nltk.download(\'punkt\')`'
            )
            return None, None, None

        scores_rouge1 = max(score['rouge-1']['r'] for score in scores)
        scores_rouge2 = max(score['rouge-2']['r'] for score in scores)
        scores_rougeL = max(score['rouge-l']['r'] for score in scores)
        return (
            RougeMetric(scores_rouge1),
            RougeMetric(scores_rouge2),
            RougeMetric(scores_rougeL),
        )


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s


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
        total += task_report.get('exs', 0)
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
    Threadsafe metrics container focused on aggregation.
    """

    def __init__(self, threadsafe=False):
        self._metrics = {}
        self._threadsafe = threadsafe
        if self._threadsafe:
            # Threadsafe metrics tracking works by keeping a queue that workers can
            # push updates to. the main worker works through the queue at report
            # time. We could add some buffering to improve performance, but we
            # are deprioritizing hogwild performance at this time.
            self._queue = multiprocessing.Queue()

    def __str__(self):
        return str(self._metrics)

    def __repr__(self):
        return f'Metrics({repr(self._metrics)})'

    def add(self, key: str, value: Optional[Metric]) -> None:
        """
        Record an accumulation to a metric.
        """
        if self._threadsafe:
            self._queue.put((key, value))
        else:
            self._metrics[key] = self._metrics.get(key) + value

    def report(self):
        """
        Report the metrics over all data seen so far.
        """
        self._sync()
        return {
            k: v.value() if isinstance(v, Metric) else v
            for k, v in self._metrics.items()
        }

    def _sync(self):
        """
        Process all items on the queue to ensure it is up to date.
        """
        for key, value in self._drain_queue():
            self._metrics[key] = self._metrics.get(key) + value

    def _drain_queue(self):
        """
        Drain the queue, yielding all items in it.
        """
        if not self._threadsafe:
            return
        while not self._queue.empty():
            try:
                yield self._queue.get()
            except queue.Empty:
                break

    def clear(self):
        """
        Clear all the metrics.
        """
        for _ in self._drain_queue():
            pass
        self._metrics.clear()


class TeacherMetrics(Metrics):
    """
    Helper container which encapsulates standard metrics (F1, BLEU, ...).
    """

    def __init__(self, threadsafe: bool = False, metrics_list: str = "default") -> None:
        super().__init__(threadsafe=threadsafe)
        self._metrics_list = self._infer_metrics(metrics_list)
        self.eval_pr = [1, 5, 10, 100]

    @staticmethod
    def _infer_metrics(cli_arg: str) -> Set[str]:
        """
        Parse the CLI metric into a list of metrics we wish to compute.
        """
        col: Set[str] = set()
        names = cli_arg.split(",")
        for n in names:
            if n == 'default':
                col |= DEFAULT_METRICS
            elif n == 'rouge':
                col |= ROUGE_METRICS
            elif n == 'bleu':
                col |= BLEU_METRICS
            elif n == 'all':
                col |= ALL_METRICS
            else:
                col.add(n)
        return col

    def _update_ranking_metrics(self, observation, labels):
        text_cands = observation.get('text_candidates', None)
        if text_cands is None:
            return

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
        for k in self.eval_pr:
            self.add(f'hits@{k}', AverageMetric(cnts[k] > 0))

    def evaluate_response(self, observation: Message, labels: List[str]) -> None:
        """
        Compute all required text-based metrics based on an observation and labels.
        """
        prediction = observation.get('text', None)

        self.add('exs', SumMetric(1))

        if prediction is not None:
            self.add('accuracy', ExactMatchMetric.compute(prediction, labels))
            self.add('f1', F1Metric.compute(prediction, labels))

            for k in range(1, 5):  # 1..4
                if f'bleu-{k}' in self._metrics_list:
                    self.add(f'bleu-{k}', BleuMetric.compute(prediction, labels, k))
            # if any of the rouges are in the list
            if self._metrics_list & ROUGE_METRICS:
                r1, r2, rL = RougeMetric.compute_many(prediction, labels)
                if 'rouge-1' in self._metrics_list:
                    self.add('rouge-1', r1)
                if 'rouge-2' in self._metrics_list:
                    self.add('rouge-2', r2)
                if 'rouge-L' in self._metrics_list:
                    self.add('rouge-L', rL)

        # Ranking metrics.
        self._update_ranking_metrics(observation, labels)

        # User-reported metrics
        if 'metrics' in observation:
            for uk, v in observation['metrics'].items():
                if uk in ALL_METRICS:
                    # don't let the user override our metrics
                    uk = f'USER_{uk}'
                assert isinstance(uk, str), type(k)
                if not isinstance(v, Metric):
                    warn_once(f'Metric {uk} is assumed to be averaged per example.')
                    v = AverageMetric(v)
                assert isinstance(v, Metric)
                self.add(uk, v)

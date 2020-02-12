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
import queue
import functools
from typing import Union, List, Optional, Tuple, Set, Any, Dict

import torch

from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector

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
    from fairseq import bleu as fairseqbleu
except ImportError:
    fairseqbleu = None

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

    @abstractmethod
    def __add__(self, other: Any) -> 'Metric':
        raise NotImplementedError

    def __iadd__(self, other):
        return self.__radd__(other)

    def __radd__(self, other: Any):
        if other is None:
            return self
        return self.__add__(other)

    def __str__(self) -> str:
        return f'{self.value():.4g}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value():.4g})'

    def __float__(self) -> float:
        return float(self.value())

    def __int__(self) -> int:
        return int(self.value())

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

    def __sub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__sub__ is intentionally limited to floats.')
        return self.value() - other

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

    @classmethod
    def many(cls, *objs: List[TVector]) -> List['Metric']:
        """
        Construct many of a Metric from the base parts.

        Useful if you separately compute numerators and denomenators, etc.
        """
        lengths = [len(o) for o in objs]
        if len(set(lengths)) != 1:
            raise IndexError(f'Uneven {cls.__name__} constructions: {lengths}')
        return [cls(*items) for items in zip(*objs)]


class FixedMetric(Metric):
    """
    Fixed metrics are verified to be the same when combined, or throw an error.
    """

    __slots__ = ('_value',)

    def __init__(self, value: TScalar):
        self._value = self.as_number(value)

    def __add__(self, other: Optional['FixedMetric']) -> 'FixedMetric':
        if other is None:
            return self
        if self != other:
            raise ValueError(f"FixedMetrics not the same: {self} and {other}")
        return self

    def value(self) -> float:
        return self._value


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
        # always keep the same return type
        return type(self)(sum_=full_sum)

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
        # always keep the same return type
        return type(self)(numer=full_numer, denom=full_denom)

    def value(self) -> float:
        if self._numer == 0 and self._denom == 0:
            # don't nan out if we haven't counted anything
            return 0.0
        if self._denom == 0:
            return float('nan')
        return self._numer / self._denom


class LegacyMetric(AverageMetric):
    """
    Legacy Metrics are reported by agent as float.
    """

    pass


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


class FairseqBleuMetric(AverageMetric):
    @staticmethod
    def compute_many(
        guess: torch.Tensor, answers: torch.Tensor, pad_idx, end_idx, unk_idx
    ):
        """
        Return BLEU-1..4 using fairseq and tokens.
        """
        if fairseqbleu is None:
            return None
        scorer = fairseqbleu.Scorer(pad_idx, end_idx, unk_idx)
        answers = answers.cpu().int()
        guess = guess.cpu().int()
        scorer.add(answers, guess)
        return [FairseqBleuMetric(scorer.score(i) / 100.0) for i in range(1, 5)]


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


def aggregate_named_reports(named_reports: Dict[str, Dict[str, Metric]]):
    """
    Aggregate metrics from multiple reports.

    :param reports: Dict of tasks -> metrics.
    """
    # reporters is a list of teachers or worlds
    m: Dict[str, Metric] = {}
    for task_id, task_report in named_reports.items():
        for each_metric, value in task_report.items():
            m[each_metric] = m.get(each_metric, None) + value
            if len(named_reports) > 1:
                m[f'{task_id}/{each_metric}'] = value
    return m


def aggregate_unnamed_reports(reports: List[Dict[str, Metric]]):
    """
    Combines metrics without regard for tracking provenence.
    """
    m: Dict[str, Metric] = {}
    for task_report in reports:
        for each_metric, value in task_report.items():
            m[each_metric] = m.get(each_metric) + value
    return m


class Metrics(object):
    """
    Threadsafe metrics container focused on aggregation.
    """

    def __init__(self, threadsafe=False, shared=None):
        self._threadsafe = threadsafe
        if self._threadsafe and shared is None:
            # Threadsafe metrics tracking works by keeping a queue that workers can
            # push updates to. the main worker works through the queue at report
            # time. We could add some buffering to improve performance, but we
            # are deprioritizing hogwild performance at this time.
            self._buffer = None
            self._queue = multiprocessing.SimpleQueue()
            self._worker = False
            self._data = {}
        elif shared and 'queue' in shared:
            # This is a clone, in threadsafe mode
            self._buffer = {}
            self._queue = shared['queue']
            self._worker = True
            self._data = None
        elif shared and 'data' in shared:
            # This is a clone, in non-threadsafe mode
            self._buffer = None
            self._queue = None
            self._worker = False
            self._data = shared['data']
        else:
            # The original in non-threadsafe mode
            self._buffer = None
            self._queue = None
            self._worker = False
            self._data = {}

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f'Metrics({repr(self._data)})'

    def add(self, key: str, value: Optional[Metric]) -> None:
        """
        Record an accumulation to a metric.
        """
        if self._threadsafe and self._worker:
            self._buffer[key] = self._buffer.get(key) + value
        else:
            self._data[key] = self._data.get(key) + value

    def flush(self):
        """
        Clear the local buffer and push it on.
        """
        if self._threadsafe and self._buffer:
            self._queue.put(self._buffer)
            self._buffer.clear()

    def report(self):
        """
        Report the metrics over all data seen so far.
        """
        self.sync()
        return {k: v for k, v in self._data.items()}

    def sync(self):
        """
        Process all items on the queue to ensure it is up to date.
        """
        if self._worker:
            self.flush()
        elif self._threadsafe and not self._worker:
            for buffer_ in self._drain_queue():
                for key, value in buffer_.items():
                    self._data[key] = self._data.get(key) + value

    def _drain_queue(self):
        """
        Drain the queue, yielding all items in it.
        """
        while not self._queue.empty():
            try:
                yield self._queue.get()
            except queue.Empty:
                break

    def clear(self):
        """
        Clear all the metrics.
        """
        if self._worker:
            self._buffer.clear()
        elif self._threadsafe and not self._worker:
            for _ in self._drain_queue():
                pass
        if self._data:
            self._data.clear()

    def share(self):
        if self._threadsafe:
            return {'queue': self._queue}
        else:
            return {'data': self._data}


class TeacherMetrics(Metrics):
    """
    Helper container which encapsulates standard metrics (F1, BLEU, ...).
    """

    def __init__(
        self,
        threadsafe: bool = False,
        metrics_list: str = "default",
        shared: Dict[str, Any] = None,
    ) -> None:
        super().__init__(threadsafe=threadsafe, shared=shared)
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

        # always flush at the end of processing this response
        self.flush()

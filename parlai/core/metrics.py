#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Provides standard metric evaluations for dialog, as well as an aggregator.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
import functools
import datetime
import math
from typing import (
    Any,
    Counter as TCounter,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch

from parlai.core.message import Message
from parlai.utils.misc import warn_once
from parlai.utils.typing import TScalar, TVector

DEFAULT_METRICS = {'bleu-4', 'accuracy', 'f1'}
ROUGE_METRICS = {'rouge-1', 'rouge-2', 'rouge-L'}
BLEU_METRICS = {'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4'}
DISTINCT_METRICS = {
    'interdistinct-1',
    'interdistinct-2',
    'intradistinct-1',
    'intradistinct-2',
}
ALL_METRICS = DEFAULT_METRICS | ROUGE_METRICS | BLEU_METRICS | DISTINCT_METRICS


class MetricDisplayData(NamedTuple):
    title: str
    description: str


METRICS_DISPLAY_DATA = {
    "accuracy": MetricDisplayData("Accuracy", "Exact match text accuracy"),
    'auc': MetricDisplayData(
        'AUC',
        "Area Under the Receiver Operating Characteristic Curve (true positive rate vs false positive rate curve)",
    ),
    "bleu-4": MetricDisplayData(
        "BLEU-4",
        "BLEU-4 of the generation, under a standardized (model-independent) tokenizer",
    ),
    "clen": MetricDisplayData(
        "Context Length", "Average length of context in number of tokens"
    ),
    "clip": MetricDisplayData(
        "Clipped Gradients", "Fraction of batches with clipped gradients"
    ),
    "ctpb": MetricDisplayData("Context Tokens Per Batch", "Context tokens per batch"),
    "ctps": MetricDisplayData("Context Tokens Per Second", "Context tokens per second"),
    "ctrunc": MetricDisplayData(
        "Context Truncation", "Fraction of samples with some context truncation"
    ),
    "ctrunclen": MetricDisplayData(
        "Context Truncation Length", "Average length of context tokens truncated"
    ),
    "exps": MetricDisplayData("Examples Per Second", "Examples per second"),
    "exs": MetricDisplayData(
        "Examples", "Number of examples processed since last print"
    ),
    "f1": MetricDisplayData(
        "F1", "Unigram F1 overlap, under a standardized (model-independent) tokenizer"
    ),
    "gen_n_toks": MetricDisplayData(
        "Generation Length", "Average length of generated outputs in number of tokens"
    ),
    "gnorm": MetricDisplayData("Gradient Norm", "Gradient norm"),
    "gpu_mem": MetricDisplayData(
        "GPU Memory",
        "Fraction of GPU memory used. May slightly underestimate true value.",
    ),
    "hits@1": MetricDisplayData(
        "Hits@1", "Fraction of correct choices in 1 guess. (Similar to recall@K)"
    ),
    "hits@5": MetricDisplayData(
        "Hits@5", "Fraction of correct choices in 5 guesses. (Similar to recall@K)"
    ),
    "interdistinct-1": MetricDisplayData(
        "Interdistinct-1", "Fraction of n-grams unique across _all_ generations"
    ),
    "interdistinct-2": MetricDisplayData(
        "Interdistinct-1", "Fraction of n-grams unique across _all_ generations"
    ),
    "intradistinct-1": MetricDisplayData(
        "Intradictinct-1", "Fraction of n-grams unique _within_ each utterance"
    ),
    "intradictinct-2": MetricDisplayData(
        "Intradictinct-2", "Fraction of n-grams unique _within_ each utterance"
    ),
    "jga": MetricDisplayData("Joint Goal Accuracy", "Joint Goal Accuracy"),
    "llen": MetricDisplayData(
        "Label Length", "Average length of label in number of tokens"
    ),
    "loss": MetricDisplayData("Loss", "Loss"),
    "lr": MetricDisplayData("Learning Rate", "The most recent learning rate applied"),
    "ltpb": MetricDisplayData("Label Tokens Per Batch", "Label tokens per batch"),
    "ltps": MetricDisplayData("Label Tokens Per Second", "Label tokens per second"),
    "ltrunc": MetricDisplayData(
        "Label Truncation", "Fraction of samples with some label truncation"
    ),
    "ltrunclen": MetricDisplayData(
        "Label Truncation Length", "Average length of label tokens truncated"
    ),
    "rouge-1": MetricDisplayData("ROUGE-1", "ROUGE metrics"),
    "rouge-2": MetricDisplayData("ROUGE-2", "ROUGE metrics"),
    "rouge-L": MetricDisplayData("ROUGE-L", "ROUGE metrics"),
    "token_acc": MetricDisplayData(
        "Token Accuracy", "Token-wise accuracy (generative only)"
    ),
    "token_em": MetricDisplayData(
        "Token Exact Match",
        "Utterance-level token accuracy. Roughly corresponds to perfection under greedy search (generative only)",
    ),
    "total_train_updates": MetricDisplayData(
        "Total Train Updates", "Number of SGD steps taken across all batches"
    ),
    "tpb": MetricDisplayData(
        "Tokens Per Batch", "Total tokens (context + label) per batch"
    ),
    "tps": MetricDisplayData(
        "Tokens Per Second", "Total tokens (context + label) per second"
    ),
    "ups": MetricDisplayData("Updates Per Second", "Updates per second (approximate)"),
}


def get_metric_display_data(metric: str) -> MetricDisplayData:
    return METRICS_DISPLAY_DATA.get(
        metric,
        MetricDisplayData(
            title=metric,
            description="No description provided. Please add it to metrics.py if this is an official metric in ParlAI.",
        ),
    )


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


@functools.total_ordering  # type: ignore
class Metric(ABC):
    """
    Base class for storing metrics.

    Subclasses should define .value(). Examples are provided for each subclass.
    """

    @property
    def is_global(self) -> bool:
        """
        Indicates whether this metric should be reported globally or per-task.
        """
        return False

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return False

    @abstractmethod
    def value(self) -> float:
        """
        Return the value of the metric as a float.
        """
        pass

    @abstractmethod
    def __add__(self, other: Any) -> Metric:
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

    def __rsub__(self, other: Any) -> float:
        """
        Used heavily for assertAlmostEqual.

        NOTE: This is not necessary in python 3.7+.
        """
        if not isinstance(other, float):
            raise TypeError('Metrics.__rsub__ is intentionally limited to floats.')
        return other - self.value()

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
    def many(cls, *objs: List[TVector]) -> List[Metric]:
        """
        Construct many of a Metric from the base parts.

        Useful if you separately compute numerators and denomenators, etc.
        """
        lengths = [len(o) for o in objs]
        objs = list(objs)  # convert from tuple for inplace modification
        for i, o in enumerate(objs):
            if isinstance(o, torch.Tensor):
                # if the tensor is on GPU, make sure we transfer the whole thing
                # at once, instead of one-element-at-a-time during our list
                # comprehension
                objs[i] = o.tolist()
        if len(set(lengths)) != 1:
            raise IndexError(f'Uneven {cls.__name__} constructions: {lengths}')
        return [cls(*items) for items in zip(*objs)]


class FixedMetric(Metric):
    """
    Fixed metrics are verified to be the same when combined, or throw an error.

    FixedMetric is used for things like total_train_updates, which should not be
    combined across different multitasks or different workers.
    """

    __slots__ = ('_value',)

    def __init__(self, value: TScalar):
        self._value = self.as_number(value)

    def __add__(self, other: Optional[FixedMetric]) -> FixedMetric:
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

    Examples of SumMetric include things like "exs", the number of examples seen since
    the last report, which depends exactly on a teacher.
    """

    __slots__ = ('_sum',)

    def __init__(self, sum_: TScalar = 0):
        if isinstance(sum_, torch.Tensor):
            self._sum = sum_.item()
        else:
            assert isinstance(sum_, (int, float))
            self._sum = sum_

    def __add__(self, other: Optional[SumMetric]) -> SumMetric:
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

    Examples of AverageMetrics include hits@1, F1, accuracy, etc. These metrics all have
    per-example values that can be directly mapped back to a teacher.
    """

    __slots__ = ('_numer', '_denom')

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, numer: TScalar, denom: TScalar = 1):
        self._numer = self.as_number(numer)
        self._denom = self.as_number(denom)

    def __add__(self, other: Optional[AverageMetric]) -> AverageMetric:
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


class MacroAverageMetric(Metric):
    """
    Class that represents the macro average of several numbers.

    Used for aggregating task level metrics. It is only used for things that are
    AverageMetrics already.
    """

    __slots__ = '_values'

    def __init__(self, metrics: Dict[str, Metric]) -> None:
        self._values = metrics

    def __add__(self, other: Optional[MacroAverageMetric]) -> MacroAverageMetric:
        if other is None:
            return self
        output = dict(**self._values)
        for k, v in other._values.items():
            output[k] = output.get(k, None) + v
        return MacroAverageMetric(output)

    def value(self) -> float:
        sum_ = sum(v.value() for v in self._values.values())
        n = len(self._values)
        return sum_ / n


class TimerMetric(Metric):
    """
    A timer metric keep tracks of the first/last times it was used.
    """

    __slots__ = ('_value', '_start', '_end')

    @classmethod
    def _now(cls) -> float:
        return datetime.datetime.utcnow().timestamp()

    def __init__(
        self,
        value: TScalar,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        self._value = self.as_number(value)
        if start_time is None:
            start_time = self._now()
        if end_time is None:
            end_time = self._now()
        self._start = start_time
        self._end = end_time

    def __add__(self, other: Optional[TimerMetric]) -> TimerMetric:
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        total: TScalar = self._value + other._value
        start: float = min(self._start, other._start)
        end: float = max(self._end, other._end)
        return type(self)(total, start, end)

    def value(self) -> float:
        if self._value == 0 or self._end == self._start:
            return 0
        return self._value / (self._end - self._start)


class GlobalMetric:
    """
    A global metric is one that should not be aggregated across different tasks.

    Examples of global metric include things like learning rate and updates.
    These need to be accumulated or averaged over multiple parleys, but cannot
    be correlated with a single task.

    Key to it is the notion that any one worker or any one task already has a global
    view of the value, and so no combinations should be done. Note this is different
    then a FixedMetric, in that a GlobalMetric can be still averaged across multiple
    parleys(), but a FixedMetric is always fixed.
    """

    @property
    def is_global(self) -> bool:
        return True


class GlobalFixedMetric(GlobalMetric, FixedMetric):
    """
    Global fixed metric.

    Used for things like total_train_updates.
    """

    pass


class GlobalSumMetric(GlobalMetric, SumMetric):
    """
    Global sum metric.

    Used for 'exs' and 'updates'.
    """

    pass


class GlobalAverageMetric(GlobalMetric, AverageMetric):
    """
    Global Average metric.

    Used for things like learning rate, and many agent-specific metrics.
    """

    pass


class LegacyMetric(GlobalAverageMetric):
    """
    Legacy Metrics are reported by agent as float.
    """

    pass


class GlobalTimerMetric(GlobalMetric, TimerMetric):
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
    def compute(guess: str, answers: List[str]) -> F1Metric:
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
    def compute(guess: str, answers: List[str]) -> ExactMatchMetric:
        if guess is None or answers is None:
            return None
        guess = normalize_answer(guess)
        for a in answers:
            if guess == normalize_answer(a):
                return ExactMatchMetric(1)
        return ExactMatchMetric(0)


class BleuMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str], k: int = 4) -> Optional[BleuMetric]:
        """
        Compute approximate BLEU score between guess and a set of answers.
        """
        try:
            from nltk.translate import bleu_score as nltkbleu
        except ImportError:
            # User doesn't have nltk installed, so we can't use it for bleu
            # We'll just turn off things, but we might want to warn the user
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


class FairseqBleuMetric(Metric):
    """
    Re-implementation of
    https://github.com/pytorch/fairseq/blob/main/fairseq/scoring/bleu.py.
    """

    def __init__(
        self,
        pred: Union[torch.Tensor, List[int]],
        ref: Union[torch.Tensor, List[int]],
        pad_idx: int,
        eos_idx: int,
        unk_idx: int,
        order: int,
    ):
        try:
            from fairseq import libbleu
            from fairseq.scoring.bleu import BleuStat
            import ctypes
        except ImportError:
            return

        self.stat = BleuStat()
        self.order = order

        C = ctypes.cdll.LoadLibrary(libbleu.__file__)
        C.bleu_zero_init(ctypes.byref(self.stat))

        if not torch.is_tensor(pred):
            pred = torch.LongTensor(pred)
        if not torch.is_tensor(ref):
            ref = torch.LongTensor(ref)

        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(unk_idx)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(pad_idx),
            ctypes.c_int(eos_idx),
        )

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __add__(self, other: Optional[FairseqBleuMetric]) -> FairseqBleuMetric:
        if other is None:
            return self
        self.stat.match1 += other.stat.match1
        self.stat.match2 += other.stat.match2
        self.stat.match3 += other.stat.match3
        self.stat.match4 += other.stat.match4
        self.stat.count1 += other.stat.count1
        self.stat.count2 += other.stat.count2
        self.stat.count3 += other.stat.count3
        self.stat.count4 += other.stat.count4
        self.stat.predlen += other.stat.predlen
        self.stat.reflen += other.stat.reflen
        return self

    def _ratio(self, a: int, b: int) -> float:
        """
        Safe division.
        """
        return a / b if b > 0 else 0

    def _precision(self):
        return [
            self._ratio(self.stat.match1, self.stat.count1),
            self._ratio(self.stat.match2, self.stat.count2),
            self._ratio(self.stat.match3, self.stat.count3),
            self._ratio(self.stat.match4, self.stat.count4),
        ]

    def _brevity(self):
        r = self.stat.reflen / self.stat.predlen
        return min(1, math.exp(1 - r))

    def value(self) -> float:
        """
        Reimplementation of Fairseq's score.
        """
        psum = sum(
            math.log(p) if p > 0 else float("-Inf")
            for p in self._precision()[: self.order]
        )
        return self._brevity() * math.exp(psum / self.order) * 100

    @staticmethod
    def compute_many(
        guess: torch.Tensor, answers: torch.Tensor, pad_idx, end_idx, unk_idx
    ):
        """
        Return BLEU-1..4 using fairseq and tokens.
        """
        try:
            from fairseq.scoring import bleu as fairseqbleu  # noqa
        except ImportError:
            return None

        return [
            FairseqBleuMetric(
                guess.cpu().int(),
                answers.cpu().int(),
                pad_idx,
                end_idx,
                unk_idx,
                order=i,
            )
            for i in range(1, 5)
        ]


class RougeMetric(AverageMetric):
    _evaluator = None

    @staticmethod
    def compute_many(
        guess: str, answers: List[str]
    ) -> Tuple[Optional[RougeMetric], Optional[RougeMetric], Optional[RougeMetric]]:
        """
        Compute ROUGE score between guess and *any* answer.

        Done with compute_many due to increased efficiency.

        :return: (rouge-1, rouge-2, rouge-L)
        """
        # possible global initialization
        try:
            import rouge
        except ImportError:
            # User doesn't have py-rouge installed, so we can't use it.
            # We'll just turn off rouge computations
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


class IntraDistinctMetric(AverageMetric):
    """
    Compute intra-distinct (per-utterance).
    """

    @classmethod
    def _ngram(cls, seq, n: int):
        for i in range(len(seq) - n + 1):
            yield tuple(seq[i : i + n])

    @classmethod
    def compute(cls, text: str, ngram: int = 1):
        """
        :param text:
            The text to compute metric over
        :param ngram:
            n-gram length
        """
        tokens = normalize_answer(text).split()
        counts: Counter[Any] = Counter(cls._ngram(tokens, ngram))
        # computed per-example, macro averaged across examples
        intra = max(len(counts), 1e-12) / max(sum(counts.values()), 1e-5)
        return IntraDistinctMetric(intra, 1.0)


class InterDistinctMetric(Metric):
    """
    Compute inter-distinct metric over corpus-level.
    """

    def __init__(self, counts: TCounter[Tuple]):
        """
        :param counts:
            collections.Counter of ngram -> frequency
        """
        self._counts = counts

    def __add__(self, other):
        return InterDistinctMetric(self._counts + other._counts)

    def value(self):
        return max(len(self._counts), 1e-12) / max(sum(self._counts.values()), 1e-5)

    @classmethod
    def _ngram(cls, seq, n):
        for i in range(len(seq) - n + 1):
            yield tuple(seq[i : i + n])

    @classmethod
    def compute(cls, text, ngram=1):
        tokens = normalize_answer(text).split()
        return InterDistinctMetric(Counter(cls._ngram(tokens, ngram)))


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


def aggregate_named_reports(
    named_reports: Dict[str, Dict[str, Metric]], micro_average: bool = False
) -> Dict[str, Metric]:
    """
    Aggregate metrics from multiple reports.

    :param reports:
        Dict of tasks -> metrics.
    :param micro_average:
        If true, top level metrics will be the micro average. By default, we
        use macro average.
    :return:
        The aggregated report
    """
    if len(named_reports) == 0:
        raise ValueError("Cannot aggregate empty reports.")
    if len(named_reports) == 1:
        # no real aggregation to be done
        return next(iter(named_reports.values()))

    # reporters is a list of teachers or worlds
    m: Dict[str, Metric] = {}
    macro_averages: Dict[str, Dict[str, Metric]] = {}
    for task_id, task_report in named_reports.items():
        for each_metric, value in task_report.items():
            if value.is_global:
                # just take the first one we saw
                if each_metric not in m:
                    m[each_metric] = value
            else:
                task_metric = f'{task_id}/{each_metric}'
                m[task_metric] = m.get(task_metric) + value
                if micro_average or not value.macro_average:
                    # none + a => a from implementation of Metric.__add__
                    m[each_metric] = m.get(each_metric) + value
                else:
                    # macro average
                    if each_metric not in macro_averages:
                        macro_averages[each_metric] = {}
                    macro_averages[each_metric][task_id] = value
    for key, values in macro_averages.items():
        m[key] = MacroAverageMetric(values)
    return m


def aggregate_unnamed_reports(reports: List[Dict[str, Metric]]) -> Dict[str, Metric]:
    """
    Combines metrics without regard for tracking provenence.
    """
    m: Dict[str, Metric] = {}
    for task_report in reports:
        for each_metric, value in task_report.items():
            m[each_metric] = m.get(each_metric) + value
    return m


def dict_report(report: Dict[str, Metric]):
    return {k: v.value() if isinstance(v, Metric) else v for k, v in report.items()}


class Metrics(object):
    """
    Metrics aggregator.
    """

    def __init__(self, threadsafe=False, shared=None):
        if shared and 'data' in shared:
            # This is a clone
            self._data = shared['data']
        else:
            # The original
            self._data = {}

        # recent data is to track per-example metrics, and so should never be
        # shared
        self._recent_data = {}

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f'Metrics({repr(self._data)})'

    def add(self, key: str, value: Optional[Metric]) -> None:
        """
        Record an accumulation to a metric.
        """
        self._data[key] = self._data.get(key) + value
        self._recent_data[key] = self._recent_data.get(key) + value

    def report(self):
        """
        Report the metrics over all data seen so far.
        """
        return self._data.copy()

    def clear_recent(self):
        """
        Clear recent metrics (latest example).
        """
        self._recent_data.clear()

    def report_recent(self):
        """
        Report recent metrics (latest example).
        """
        return self._recent_data.copy()

    def clear(self):
        """
        Clear all the metrics.
        """
        self._data.clear()
        self._recent_data.clear()

    def share(self):
        return {'data': self._data}

    def add_metrics(self, other: "Metrics") -> None:
        """
        Aggregate another Metrics objects metrics into this one.

        Note that it is assumed that the keys for metrics are disjoint between Metrics
        objects.
        """
        for k, v in other._data.items():
            self.add(k, v)


class TeacherMetrics(Metrics):
    """
    Helper container which encapsulates standard metrics (F1, BLEU, ...).
    """

    def __init__(
        self, metrics_list: str = "default", shared: Dict[str, Any] = None
    ) -> None:
        super().__init__(shared=shared)
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
            elif n == 'distinct':
                col |= DISTINCT_METRICS
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
                if 'rouge-1' in self._metrics_list and r1:
                    self.add('rouge_1', r1)
                if 'rouge-2' in self._metrics_list and r2:
                    self.add('rouge_2', r2)
                if 'rouge-L' in self._metrics_list and rL:
                    self.add('rouge_L', rL)
            # compute distinct-k
            for k in [1, 2]:
                if f'interdistinct-{k}' in self._metrics_list:
                    self.add(
                        f'interdistinct-{k}', InterDistinctMetric.compute(prediction, k)
                    )
                if f'intradistinct-{k}' in self._metrics_list:
                    self.add(
                        f'intradistinct-{k}', IntraDistinctMetric.compute(prediction, k)
                    )

        # Ranking metrics.
        self._update_ranking_metrics(observation, labels)

        self._consume_user_metrics(observation)

    def _consume_user_metrics(self, observation):
        # User-reported metrics
        if 'metrics' in observation:
            for uk, v in observation['metrics'].items():
                if uk in ALL_METRICS:
                    # don't let the user override our metrics
                    uk = f'USER_{uk}'
                assert isinstance(uk, str), f'{type(uk)} is not a str'
                if not isinstance(v, Metric):
                    warn_once(f'Metric {uk} is assumed to be averaged per example.')
                    v = AverageMetric(v)
                assert isinstance(v, Metric)
                self.add(uk, v)

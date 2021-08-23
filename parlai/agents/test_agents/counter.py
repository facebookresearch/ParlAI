#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test agent which counts its number of unique items.
"""

from __future__ import annotations
from typing import Tuple
from collections import Counter

from parlai.core.torch_agent import TorchAgent
from parlai.core.metrics import Metric, SumMetric
from parlai.core.message import Message


class _CounterMetric(Metric):
    __slots__ = ('_counter',)

    def __init__(self, counter: Counter):
        self._counter = counter

    def __add__(self, other: Metric):
        if other is None:
            return self
        counter = self._counter + other._counter
        return type(self)(counter)


class TimesSeenMetric(_CounterMetric):
    """
    Max number of times any example was seen.
    """

    def value(self) -> int:
        if not self._counter:
            return 0
        return max(self._counter.values())


class UniqueMetric(_CounterMetric):
    """
    Number of unique utterances.
    """

    def value(self) -> int:
        if not self._counter:
            return 0
        return len(self._counter)


class FakeList(list):
    @property
    def batchsize(self):
        return len(self)


class CounterAgent(TorchAgent):
    """
    Simple agent that counts the number of unique things it has seen.

    Could be simpler, but we make it a TorchAgent so it's happy with dynamic batching.
    """

    def __init__(self, opt, shared=None):
        self.model = self.build_model()
        self.criterion = None
        super().__init__(opt, shared)
        self._counter: Counter[Tuple[str, str]]
        if shared is None:
            self._counter = Counter()
            self._padding_counter = Counter()
        else:
            self._counter = shared['counter']
            self._padding_counter = shared['padding']

    def share(self):
        shared = super().share()
        shared['counter'] = self._counter
        shared['padding'] = self._padding_counter
        return shared

    def save(self, path=None):
        pass

    def load(self, path=None):
        pass

    def _val(self, val):
        """
        Pull out a singleton value if provided a list.
        """
        # necessary for labels
        if isinstance(val, (tuple, list)):
            return val[0]
        else:
            return val

    def build_model(self):
        return None

    def train_step(self):
        pass

    def eval_step(self):
        pass

    def batchify(self, observations):
        return FakeList(observations)

    def _to_tuple(self, msg: Message) -> Tuple:
        # turned into an indexable object
        keys = ['text', 'labels', 'eval_labels']
        return tuple(self._val(msg.get(k)) for k in keys)

    def batch_act(self, observations):
        self._padding_counter.update(['val' for o in observations if o.is_padding()])
        self._counter.update(
            [self._to_tuple(o) for o in observations if not o.is_padding()]
        )
        return [Message() for o in observations]

    def reset(self):
        self._counter.clear()

    def report(self):
        report = {}
        report['num_pad'] = SumMetric(self._padding_counter.get('val', 0))
        report['unique'] = UniqueMetric(self._counter)
        report['times_seen'] = TimesSeenMetric(self._counter)
        return report

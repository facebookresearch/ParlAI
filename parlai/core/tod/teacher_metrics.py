#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Task Oriented Dialogue (TOD) teacher metrics.
"""
from typing import Optional, List, Dict, Any
from parlai.core.metrics import AverageMetric, BleuMetric, F1Metric, Metric, Metrics


class SlotMetrics(Metrics):
    """
    Helper container which encapsulates standard slot metrics in task oriented learning
    (jga, slot_p, slot_r, etc).

    Due to differences in dialogue representations between tasks, the input is pre-
    parsed ground truth and predicted slot dictionaries.
    """

    def __init__(
        self,
        teacher_slots: Dict[str, str],
        predicted_slots: Dict[str, str],
        prefixes: Optional[List] = None,
        shared: Dict[str, Any] = None,
    ) -> None:
        super().__init__(shared=shared)
        self.prefixes = prefixes if prefixes else []
        self.add_with_prefixes("jga", AverageMetric(teacher_slots == predicted_slots))
        if len(teacher_slots) > 0:
            self.add_with_prefixes(
                "jga_noempty", AverageMetric(teacher_slots == predicted_slots)
            )
        else:
            self.add_with_prefixes(
                "jga_empty", AverageMetric(teacher_slots == predicted_slots)
            )

        # precision
        for pred_slot_name, pred_value in predicted_slots.items():
            slot_p = AverageMetric(teacher_slots.get(pred_slot_name) == pred_value)
            self.add_with_prefixes("slot_p", slot_p)
            self.add_with_prefixes("slot_f1", SlotF1Metric(slot_p=slot_p))
        # recall
        for teacher_slot_name, teacher_value in teacher_slots.items():
            slot_r = AverageMetric(
                predicted_slots.get(teacher_slot_name) == teacher_value
            )
            self.add_with_prefixes("slot_r", slot_r)
            self.add_with_prefixes("slot_f1", SlotF1Metric(slot_r=slot_r))

    def add_with_prefixes(self, name, value):
        self.add(name, value)
        for prefix in self.prefixes:
            self.add(f"{prefix}/{name}", value)


class NlgMetrics(Metrics):
    """
    Helper container for generation version of standard metrics (F1, BLEU, ..).
    """

    def __init__(
        self,
        guess: str,
        labels: Optional[List[str]],
        prefixes: Optional[List[str]] = None,
        shared: Dict[str, Any] = None,
    ) -> None:
        super().__init__(shared=shared)
        self.prefixes = prefixes if prefixes else []
        bleu = BleuMetric.compute(guess, labels)
        f1 = F1Metric.compute(guess, labels)
        self.add_with_prefixes("nlg_bleu", bleu)
        self.add_with_prefixes("nlg_f1", f1)

    def add_with_prefixes(self, name, value):
        self.add(name, value)
        for prefix in self.prefixes:
            self.add(f"{prefix}/{name}", value)


AverageType = Optional[AverageMetric]


def _average_type_sum_helper(first: AverageType, second: AverageType) -> AverageType:
    """
    Helper to deal with Nones.

    We are "clever" in how we aggregate SlotF1Metrics (See SlotMetrics `__init__`) in
    that we add precision and recall values separately, but this means we need to handle
    None.
    """
    if first is None:
        return second
    if second is None:
        return first
    return first + second


class SlotF1Metric(Metric):
    """
    Metric to keep track of slot F1.

    Keeps track of slot precision and slot recall as running metrics.
    """

    __slots__ = ("_slot_p", "_slot_r")

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, slot_p: AverageType = None, slot_r: AverageType = None):
        if not isinstance(slot_p, AverageMetric) and slot_p is not None:
            slot_p = AverageMetric(slot_p)
        if not isinstance(slot_r, AverageMetric) and slot_r is not None:
            slot_r = AverageMetric(slot_r)
        self._slot_p = slot_p
        self._slot_r = slot_r

    def __add__(self, other: Optional["SlotF1Metric"]) -> "SlotF1Metric":
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        slot_p = _average_type_sum_helper(self._slot_p, other._slot_p)
        slot_r = _average_type_sum_helper(self._slot_r, other._slot_r)
        return type(self)(slot_p=slot_p, slot_r=slot_r)

    def value(self) -> float:
        if self._slot_p is None or self._slot_r is None:
            return float("nan")
        else:
            slot_p = self._slot_p.value()
            slot_r = self._slot_r.value()
            if slot_p == 0.0 and slot_r == 0.0:
                return float("nan")
            else:
                return 2 * (slot_p * slot_r) / (slot_p + slot_r)

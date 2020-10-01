#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Helper functions for collecting metrics for dialogue oriented tasks
from typing import Optional, Tuple, Union

from parlai.core.teachers import Teacher
from parlai.core.metrics import AverageMetric, BleuMetric, F1Metric, Metric
from parlai.utils.typing import TScalar


def __add_domain_metrics(
    domain: Optional[str] = None, valid_domains: Optional[Tuple[str]] = None
) -> bool:
    if domain is not None:
        if valid_domains is None or domain in valid_domains:
            return True
    return False


class SlotMetrics(Metric):
    """
    Helper container which encapsulates standard slot metrics in task-oriented learning (jga, slot_p, slot_r, etc). Due to differences in dialogue representations between tasks, the input is pre-parsed ground truth and predicted slot dictionaries. This class will add domain-specific versions of these metrics as well, when appropriate.
    """

    def __init__(
        teacher_slots={},
        predicted_slots={},
        domain: Optional[str] = None,
        valid_domains: Optional[Tuple[str]] = None,
    ):
        self.add("jga", AverageMetric(predictions == slots))
        for pred_slot_name, pred_value in predicted_slots.items():
            slot_p = AverageMetric(teacher_slots.get(pred_slot_name) == pred_value)
            self.add("slot_p", slot_p)
            self.add("slot_f1", SlotF1Metric(slot_p=slot_p))
            if __add_domain_metrics(domain, valid_domains):
                self.add(f"{domain}/slot_p", slot_p)
                self.add(f"{domain}/slot_f1", SlotF1Metric(slot_p=slot_p))

        for teacher_slot_name, teacher_value in slots.items():
            slot_r = AverageMetric(
                predicted_slots.get(teacher_slot_name) == teacher_value
            )
            self.add("slot_r", slot_r)
            self.add("slot_f1", SlotF1Metric(slot_r=slot_r))
            if __add_domain_metrics(domain, valid_domains):
                self.add(f"{domain}/slot_r", slot_r)
                self.add(f"{domain}/slot_f1", SlotF1Metric(slot_r=slot_r))


class NlgMetrics(Metric):
    """
    Helper container for generation version of standard metrics (F1, BLEU, ...). This class will add domain-specific versions of these classes as well, when appropriate
    """

    def __init__(
        teacher: Teacher,
        guess: str,
        labels: Optional[Tuple[str]],
        teacher_domains: Optional[Tuple[str]],
    ) -> None:
        bleu = BleuMetric.compute(guess, labels)
        f1 = F1Metric.compute(guess, labels)
        self.add("nlg_bleu", bleu)
        self.add("nlg_f1", f1)
        for domain in teacher_domains:
            self.add(f"{domain}/nlg_bleu", bleu)
            self.add(f"{domain}/nlg_f1", f1)


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

    def __init__(
        self,
        slot_p: Optional[Union[AverageMetric, TScalar]] = None,
        slot_r: Optional[Union[AverageMetric, TScalar]] = None,
    ):
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
        full_slot_p: AverageMetric = (
            None
            if self._slot_p is None and other._slot_p is None
            else self._slot_p + other._slot_p
        )
        full_slot_r: AverageMetric = (
            None
            if self._slot_r is None and other._slot_r is None
            else self._slot_r + other._slot_r
        )
        # always keep the same return type
        return type(self)(slot_p=full_slot_p, slot_r=full_slot_r)

    def value(self) -> Optional[float]:
        if self._slot_p is None or self._slot_r is None:
            return None
        else:
            slot_p = self._slot_p.value()
            slot_r = self._slot_r.value()
            if slot_p == 0.0 and slot_r == 0.0:
                return float("nan")
            else:
                return 2 * (slot_p * slot_r) / (slot_p + slot_r)

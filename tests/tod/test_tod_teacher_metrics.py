#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
File that includes tests for teacher metrics.
"""

import unittest
from math import isnan

from parlai.core.metrics import AverageMetric
from parlai.core.tod.teacher_metrics import SlotF1Metric, SlotMetrics


class TestSlotF1Metric(unittest.TestCase):
    """
    Test SlotF1Metric.
    """

    def test_slot_f1_metric_inputs(self):
        slots_p_r_and_f1 = [
            (None, None, float("nan")),
            (None, AverageMetric(0.0), float("nan")),
            (AverageMetric(0.0), AverageMetric(0.0), float("nan")),
            (AverageMetric(1), AverageMetric(1), 1.0),
            (AverageMetric(1), AverageMetric(0), 0.0),
            (AverageMetric(0.25), AverageMetric(0.75), 0.375),
        ]
        for slot_p, slot_r, slot_f1 in slots_p_r_and_f1:
            actual_slot_f1 = SlotF1Metric(slot_p=slot_p, slot_r=slot_r).value()
            if isnan(slot_f1):
                self.assertTrue(isnan(actual_slot_f1))
            else:
                self.assertEqual(slot_f1, actual_slot_f1)

    def test_slot_f1_metric_addition(self):
        a = SlotF1Metric(slot_p=1)
        b = SlotF1Metric(slot_r=0)
        c = SlotF1Metric(slot_p=AverageMetric(numer=2, denom=3), slot_r=1)
        d = a + b + c
        # Slot P should be 3/4 = 0.75; slot R should be 1/2 = 0.5
        self.assertEqual(0.6, d.value())


empty_slots = {}
basic_slots = {"a": "a_val", "b": "b_val", "c": "c_val"}
partial_slots = {"a": "a_val", "other": "other_val"}


class TestSlotMetrics(unittest.TestCase):
    def test_base_slot_metrics(self):
        cases = [
            (empty_slots, empty_slots, {"jga": 1}),
            (
                basic_slots,
                basic_slots,
                {"jga": 1, "slot_p": 1, "slot_r": 1, "slot_f1": 1},
            ),
            (
                basic_slots,
                partial_slots,
                {"jga": 0, "slot_p": 0.5, "slot_r": float(1.0 / 3), "slot_f1": 0.4},
            ),
        ]
        for teacher, predicted, result in cases:
            metric = SlotMetrics(teacher_slots=teacher, predicted_slots=predicted)
            for key in result:
                self.assertEqual(result[key], metric.report()[key])


if __name__ == "__main__":
    unittest.main()

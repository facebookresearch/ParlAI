#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from math import isnan

from parlai.core.metrics import AverageMetric
from parlai.utils.task_dialogue_helpers.metrics import SlotF1Metric


class TestSlotF1Metric(unittest.TestCase):
    """
    Test SlotF1Metric.
    """

    def test_slot_f1_metric_inputs(self):
        slots_p_r_and_f1 = [
            (None, None, None),
            (None, 0.0, None),
            (0.0, 0.0, float("nan")),
            (1, AverageMetric(1), 1.0),
            (AverageMetric(1), 0, 0.0),
            (1, 1, 1.0),
            (AverageMetric(1), AverageMetric(0), 0.0),
        ]
        for slot_p, slot_r, slot_f1 in slots_p_r_and_f1:
            actual_slot_f1 = SlotF1Metric(slot_p=slot_p, slot_r=slot_r).value()
            if slot_f1 is not None and isnan(slot_f1):
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


if __name__ == "__main__":
    unittest.main()

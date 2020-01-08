#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.metrics import AverageMetric, SumMetric

import unittest


class TestUtils(unittest.TestCase):
    def test_sum_metric_inputs(self):
        for input, output in passing_inputs_and_outputs:
            self.assertEqual(SumMetric(input).report(), output)
        for input in failing_inputs:
            self.assertRaises(AssertionError, SumMetric(input))

    def test_sum_metric_additions(self):
        for input1, input2, output in input_pairs_and_outputs:
            self.assertEqual((SumMetric(input1) + SumMetric(input2)).report(), output)

    def test_average_metric_inputs(self):
        for input, output in passing_inputs_and_outputs:
            self.assertEqual(AverageMetric(input[0], input[1]).report(), output)
        for input in failing_inputs:
            self.assertRaises(AssertionError, AverageMetric(input[0], input[1]))

    def test_average_metric_additions(self):
        for input1, input2, output in input_pairs_and_outputs:
            self.assertEqual(
                (
                    AverageMetric(input1[0], input1[1])
                    + AverageMetric(input2[0], input2[1])
                ).report(),
                output,
            )

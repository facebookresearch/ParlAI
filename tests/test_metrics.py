#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from parlai.core.metrics import AverageMetric, SumMetric


class TestUtils(unittest.TestCase):
    def test_sum_metric_inputs(self):

        passing_inputs_and_outputs = [
            (2, 2.0),
            (-5.0, -5.0),
            (torch.LongTensor([[-1]]), -1.0),
            (torch.DoubleTensor([34.68]), 34.68),
        ]
        for input_, output in passing_inputs_and_outputs:
            actual_output = SumMetric(input_).value()
            self.assertEqual(actual_output, output)
            # Value should be equal here. For the rest they'll only be equal to some
            # precision
            self.assertIsInstance(actual_output, float)

        failing_inputs = [
            ('4', AssertionError),
            ([6.8], AssertionError),
            (torch.Tensor([1, 3.8]), ValueError),  # Tensor has more than 1 element
        ]
        for input_, error in failing_inputs:
            with self.assertRaises(error):
                SumMetric(input_)

    def test_sum_metric_additions(self):

        input_pairs_and_outputs = [
            (1, 2, 3.0),
            (3, -5.0, -2.0),
            (torch.Tensor([[[4.2]]]), 3, 7.2),
            (torch.DoubleTensor([1]), torch.LongTensor([[-1]]), 0.0),
        ]
        for input1, input2, output in input_pairs_and_outputs:
            actual_output = (SumMetric(input1) + SumMetric(input2)).value()
            self.assertAlmostEqual(actual_output, output, places=6)
            self.assertIsInstance(actual_output, float)

    def test_average_metric_inputs(self):

        passing_inputs_and_outputs = [
            ((2, 4), 0.5),
            ((17.0, 10.0), 1.7),
            ((torch.Tensor([2.3]), torch.LongTensor([2])), 1.15),
            ((torch.Tensor([2.3]), torch.Tensor([2.0])), 1.15),
        ]
        for input_, output in passing_inputs_and_outputs:
            actual_output = AverageMetric(input_[0], input_[1]).value()
            self.assertAlmostEqual(actual_output, output, places=6)
            self.assertIsInstance(actual_output, float)

        failing_inputs = [
            ((2, '4'), AssertionError),
            ((17.0, 10.0000001), AssertionError),
            ((torch.Tensor([1, 1]), torch.Tensor([2])), ValueError),
            ((torch.Tensor([3.2]), torch.Tensor([4.2])), AssertionError),
        ]
        for input_, error in failing_inputs:
            with self.assertRaises(error):
                AverageMetric(input_[0], input_[1])

    def test_average_metric_additions(self):

        input_pairs_and_outputs = [
            ((2, 4), (1.5, 1), 0.7),
            (
                (torch.LongTensor([[[2]]]), torch.Tensor([4])),
                (torch.FloatTensor([1.5]), 1),
                0.7,
            ),
        ]
        for input1, input2, output in input_pairs_and_outputs:
            actual_output = (
                AverageMetric(input1[0], input1[1])
                + AverageMetric(input2[0], input2[1])
            ).value()
            self.assertAlmostEqual(actual_output, output, places=6)
            self.assertIsInstance(actual_output, float)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import random

from parlai.core.metrics import (
    AverageMetric,
    SumMetric,
    FixedMetric,
    Metrics,
    GlobalAverageMetric,
    MacroAverageMetric,
    aggregate_unnamed_reports,
    aggregate_named_reports,
)


class TestMetric(unittest.TestCase):
    """
    Test individual Metric classes.
    """

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
            (1, 2, 3),
            (3, -5.0, -2.0),
            (torch.Tensor([[[4.2]]]), 3, 7.2),
            (torch.DoubleTensor([1]), torch.LongTensor([[-1]]), 0),
        ]
        for input1, input2, output in input_pairs_and_outputs:
            actual_output = (SumMetric(input1) + SumMetric(input2)).value()
            self.assertAlmostEqual(actual_output, output, places=6)

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
            ((torch.Tensor([1, 1]), torch.Tensor([2])), ValueError),
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

    def test_fixedmetric(self):
        assert (FixedMetric(3) + FixedMetric(3)).value() == 3
        with self.assertRaises(ValueError):
            _ = FixedMetric(3) + FixedMetric(4)

    def test_macroaverage_additions(self):
        m1 = AverageMetric(1, 3)
        m2 = AverageMetric(3, 4)

        assert (m1 + m2) == AverageMetric(4, 7)
        assert MacroAverageMetric({'a': m1, 'b': m2}) == 0.5 * (1.0 / 3 + 3.0 / 4)


class TestMetrics(unittest.TestCase):
    """
    Test the Metrics aggregator.
    """

    def test_simpleadd(self):
        m = Metrics(threadsafe=False)
        m.add('key', SumMetric(1))
        m.add('key', SumMetric(2))
        assert m.report()['key'] == 3

        m.clear()
        assert 'key' not in m.report()

        m.add('key', SumMetric(1.5))
        m.add('key', SumMetric(2.5))
        assert m.report()['key'] == 4.0

        # shouldn't throw exception
        m.flush()

    def test_shared(self):
        m = Metrics(threadsafe=False)
        m2 = Metrics(threadsafe=False, shared=m.share())
        m3 = Metrics(threadsafe=False, shared=m.share())

        m2.add('key', SumMetric(1))
        m3.add('key', SumMetric(2))
        m2.flush()  # just make sure this doesn't throw exception, it's a no-op
        m.add('key', SumMetric(3))

        assert m.report()['key'] == 6

        # shouldn't throw exception
        m.flush()
        m2.flush()
        m3.flush()

    def test_multithreaded(self):
        m = Metrics(threadsafe=True)
        m2 = Metrics(threadsafe=True, shared=m.share())
        m3 = Metrics(threadsafe=True, shared=m.share())

        m2.add('key', SumMetric(1))
        m2.flush()
        m3.add('key', SumMetric(2))
        m3.flush()
        m.add('key', SumMetric(3))
        m.flush()
        m.report()['key'] == 6

    def test_verymultithreaded(self):
        m = Metrics(threadsafe=True)
        nt = 128
        ms = [Metrics(threadsafe=True, shared=m.share()) for _ in range(nt)]

        # intentionally just over the int overflow
        for _ in range(32768 + 1):
            ms[random.randint(0, nt - 1)].add('key', SumMetric(1))
        thread_ids = list(range(nt))
        random.shuffle(thread_ids)
        for tid in thread_ids:
            ms[tid].flush()

        assert m.report()['key'] == 32768 + 1

    def test_largebuffer(self):
        m = Metrics(threadsafe=True)
        m2 = Metrics(threadsafe=True, shared=m.share())

        # intentionally just over the int overflow
        for _ in range(32768 + 1):
            m2.add('key', SumMetric(1))
        m2.flush()

        assert m.report()['key'] == 32768 + 1


class TestAggregators(unittest.TestCase):
    def test_unnamed_aggregation(self):
        report1 = {
            'avg': AverageMetric(3, 4),
            'sum': SumMetric(3),
            'fixed': FixedMetric(4),
            'global_avg': GlobalAverageMetric(3, 4),
        }
        report2 = {
            'avg': AverageMetric(1, 3),
            'sum': SumMetric(4),
            'fixed': FixedMetric(4),
            'global_avg': GlobalAverageMetric(1, 3),
        }
        agg = aggregate_unnamed_reports([report1, report2])
        assert agg['avg'] == 4.0 / 7
        assert agg['sum'] == 7
        assert agg['fixed'] == 4
        assert agg['global_avg'] == 4.0 / 7

    def test_macro_aggregation(self):
        report1 = {
            'avg': AverageMetric(3, 4),
            'sum': SumMetric(3),
            'fixed': FixedMetric(4),
            'global_avg': GlobalAverageMetric(3, 4),
        }
        report2 = {
            'avg': AverageMetric(1, 3),
            'sum': SumMetric(4),
            'fixed': FixedMetric(4),
            'global_avg': GlobalAverageMetric(1, 3),
        }
        agg = aggregate_named_reports({'a': report1, 'b': report2}, micro_average=False)
        assert agg['avg'] == 0.5 * (3.0 / 4 + 1.0 / 3)
        assert agg['sum'] == 7
        assert agg['fixed'] == 4
        assert agg['global_avg'] in (report1['global_avg'], report2['global_avg'])
        # task level metrics
        assert agg['a/avg'] == 3.0 / 4
        assert agg['a/sum'] == 3
        assert agg['a/fixed'] == 4
        assert 'a/global_avg' not in agg
        assert agg['b/avg'] == 1.0 / 3
        assert agg['b/sum'] == 4
        assert agg['b/fixed'] == 4
        assert 'b/global_avg' not in agg

    def test_uneven_macro_aggrevation(self):
        report1 = {
            'avg': AverageMetric(1, 1),
        }
        report2 = {
            'avg': AverageMetric(0, 1),
        }
        report3 = {
            'avg': AverageMetric(0, 1),
        }
        agg1 = aggregate_named_reports(
            {'a': report1, 'b': report2}, micro_average=False
        )
        agg2 = aggregate_named_reports({'a': {}, 'c': report3}, micro_average=False)

        agg = aggregate_unnamed_reports([agg1, agg2])
        assert agg1['avg'] == 0.5
        assert agg2['avg'] == 0.0
        assert agg['a/avg'] == 1.0
        assert agg['b/avg'] == 0.0
        assert agg['c/avg'] == 0.0
        assert agg['avg'] == 1.0 / 3

    def test_micro_aggregation(self):
        report1 = {
            'avg': AverageMetric(3, 4),
            'sum': SumMetric(3),
            'fixed': FixedMetric(4),
            'global_avg': GlobalAverageMetric(3, 4),
        }
        report2 = {
            'avg': AverageMetric(1, 3),
            'sum': SumMetric(4),
            'fixed': FixedMetric(4),
            'global_avg': GlobalAverageMetric(1, 3),
        }
        agg = aggregate_named_reports({'a': report1, 'b': report2}, micro_average=True)
        assert agg['avg'] == 4.0 / 7
        assert agg['sum'] == 7
        assert agg['fixed'] == 4
        assert agg['global_avg'] in (report1['global_avg'], report2['global_avg'])
        # task level metrics
        assert agg['a/avg'] == 3.0 / 4
        assert agg['a/sum'] == 3
        assert agg['a/fixed'] == 4
        assert 'a/global_avg' not in agg
        assert agg['b/avg'] == 1.0 / 3
        assert agg['b/sum'] == 4
        assert agg['b/fixed'] == 4
        assert 'b/global_avg' not in agg


if __name__ == '__main__':
    unittest.main()

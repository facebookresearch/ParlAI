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
    TimerMetric,
    aggregate_unnamed_reports,
    aggregate_named_reports,
)
from parlai.core.torch_classifier_agent import ConfusionMatrixMetric, WeightedF1Metric


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
        m = Metrics()
        m.add('key', SumMetric(1))
        m.add('key', SumMetric(2))
        assert m.report()['key'] == 3

        m.clear()
        assert 'key' not in m.report()

        m.add('key', SumMetric(1.5))
        m.add('key', SumMetric(2.5))
        assert m.report()['key'] == 4.0

    def test_shared(self):
        m = Metrics()
        m2 = Metrics(shared=m.share())
        m3 = Metrics(shared=m.share())

        m2.add('key', SumMetric(1))
        m3.add('key', SumMetric(2))
        m.add('key', SumMetric(3))

        assert m.report()['key'] == 6

    def test_multithreaded(self):
        # legacy test, but left because it's just another test
        m = Metrics()
        m2 = Metrics(shared=m.share())
        m3 = Metrics(shared=m.share())

        m2.add('key', SumMetric(1))
        m3.add('key', SumMetric(2))
        m.add('key', SumMetric(3))
        m.report()['key'] == 6

    def test_verymultithreaded(self):
        # legacy test, but useful all the same, for ensuring
        # metrics doesn't care about the order things are done
        m = Metrics()
        nt = 128
        ms = [Metrics(shared=m.share()) for _ in range(nt)]

        # intentionally just over the int overflow
        for _ in range(32768 + 1):
            ms[random.randint(0, nt - 1)].add('key', SumMetric(1))
        thread_ids = list(range(nt))
        random.shuffle(thread_ids)
        assert m.report()['key'] == 32768 + 1

    def test_largebuffer(self):
        # legacy test. left as just another test
        m = Metrics()
        m2 = Metrics(shared=m.share())

        # intentionally just over the int overflow
        for _ in range(32768 + 1):
            m2.add('key', SumMetric(1))

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
        report1 = {'avg': AverageMetric(1, 1)}
        report2 = {'avg': AverageMetric(0, 1)}
        report3 = {'avg': AverageMetric(0, 1)}
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

    def test_time_metric(self):
        metric = TimerMetric(10, 0, 1)
        assert metric.value() == 10
        metric = TimerMetric(10, 0, 2)
        assert metric.value() == 5
        metric2 = TimerMetric(10, 4, 5)
        # final start time 0
        # final end time 5
        # total processed = 20
        assert (metric + metric2).value() == 4

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

    def test_classifier_metrics(self):
        # We assume a batch of 16 samples, binary classification case, from 2 tasks.
        # task 1
        # confusion matrix expected, for class ok,
        # TP = 2, TN = 2, FP = 2, FN = 2
        report1 = {}
        report2 = {}
        task1_f1s = {}
        task2_f1s = {}
        classes = ['class_ok', 'class_notok']
        task1_predictions = [
            'class_ok',
            'class_ok',
            'class_ok',
            'class_ok',
            'class_notok',
            'class_notok',
            'class_notok',
            'class_notok',
        ]
        task1_gold_labels = [
            'class_ok',
            'class_ok',
            'class_notok',
            'class_notok',
            'class_ok',
            'class_ok',
            'class_notok',
            'class_notok',
        ]
        for each in classes:
            precisions, recalls, f1s = ConfusionMatrixMetric.compute_metrics(
                task1_predictions, task1_gold_labels, each
            )
            report1.update(
                {
                    f'{each}_precision': sum(precisions, None),
                    f'{each}_recall': sum(recalls, None),
                    f'{each}_f1': sum(f1s, None),
                }
            )
            task1_f1s[each] = f1s
        report1['weighted_f1'] = sum(WeightedF1Metric.compute_many(task1_f1s), None)
        # task 2, for class ok
        # TP = 3, TN = 2, FP = 2, FN = 1
        # for class not ok
        # TP = 2, TN = 3, FP = 1, FN = 2
        task2_predictions = [
            'class_ok',
            'class_ok',
            'class_ok',
            'class_ok',
            'class_ok',
            'class_notok',
            'class_notok',
            'class_notok',
        ]
        task2_gold_labels = [
            'class_ok',
            'class_ok',
            'class_notok',
            'class_notok',
            'class_ok',
            'class_ok',
            'class_notok',
            'class_notok',
        ]
        for each in classes:
            precisions, recalls, f1s = ConfusionMatrixMetric.compute_metrics(
                task2_predictions, task2_gold_labels, each
            )
            report2.update(
                {
                    f'{each}_precision': sum(precisions, None),
                    f'{each}_recall': sum(recalls, None),
                    f'{each}_f1': sum(f1s, None),
                }
            )
            task2_f1s[each] = f1s
        report2['weighted_f1'] = sum(WeightedF1Metric.compute_many(task2_f1s), None)

        agg = aggregate_named_reports(
            {'task1': report1, 'task2': report2}, micro_average=False
        )
        # task1
        assert agg['task1/class_ok_precision'] == 0.5
        assert agg['task1/class_ok_recall'] == 0.5
        assert agg['task1/class_ok_f1'] == 0.5
        # task2
        assert agg['task2/class_ok_precision'] == 3 / 5
        assert agg['task2/class_ok_recall'] == 3 / 4
        assert agg['task2/class_ok_f1'] == 2 / 3
        # task2 not ok
        assert agg['task2/class_notok_precision'] == 2 / 3
        assert agg['task2/class_notok_recall'] == 0.5
        assert agg['task2/class_notok_f1'] == 4 / 7
        # weighted f1
        assert agg['task1/weighted_f1'] == 0.5
        assert agg['task2/weighted_f1'] == (2 / 3) * 0.5 + (4 / 7) * 0.5
        # all
        assert agg['weighted_f1'] == (0.5 + (2 / 3) * 0.5 + (4 / 7) * 0.5) / 2


if __name__ == '__main__':
    unittest.main()

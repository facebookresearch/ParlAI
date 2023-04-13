#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict
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
    InterDistinctMetric,
    IntraDistinctMetric,
    FairseqBleuMetric,
)
from parlai.core.torch_classifier_agent import (
    ConfusionMatrixMetric,
    WeightedF1Metric,
    AUCMetrics,
)
from parlai.core.torch_generator_agent import PPLMetric
import parlai.utils.testing as testing_utils


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
            (torch.Tensor([1, 3.8]), RuntimeError),  # Tensor has more than 1 element
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
            ((torch.Tensor([1, 1]), torch.Tensor([2])), RuntimeError),
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

    def test_average_metric_from_mask(self) -> None:
        # first test case. batchsize=3, num_tokens=10
        token_values_1 = torch.FloatTensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8],
                [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            ]
        )
        token_mask_1 = torch.LongTensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            ]
        )
        output_1 = [
            AverageMetric(55, 10),
            AverageMetric(-30, 6),
            AverageMetric(12.5, 5),
        ]

        # second test case. batchsize=4, num_tokens=5
        token_values_2 = torch.FloatTensor(
            [
                [1, 2, 3, 4, 5],
                [1.5, 0, -1, 3, -4],
                [-3, -2, -1, 0, 1],
                [4, 5, 6, 7, 8],
            ]
        )
        token_mask_2 = torch.LongTensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        )
        output_2 = [
            AverageMetric(15, 5),
            AverageMetric(0.5, 3),
            AverageMetric(-3, 3),
            AverageMetric(0, 0),
        ]

        input_and_outputs = [
            (token_values_1, token_mask_1, output_1),
            (token_values_2, token_mask_2, output_2),
        ]

        for token_values, token_mask, output in input_and_outputs:
            actual_output = AverageMetric.from_mask(token_values, token_mask)
            self.assertEqual(len(actual_output), len(output))
            # Because Metric.from_mask() calls Metric.many(), which in turn converts tensors to lists,
            # it possible for the actual and expected outputs to be close to each other but not exactly equal.
            for a, o in zip(actual_output, output):
                self.assertIsInstance(a, type(o))
                self.assertAlmostEqual(a.value(), o.value(), places=6)

    def test_ppl_metric_from_mask(self) -> None:
        # batchsize=3, num_tokens=10
        token_values = torch.FloatTensor(
            [
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            ]
        )
        token_mask = torch.LongTensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        output = [
            PPLMetric(4.5, 10),
            PPLMetric(0.6, 6),
            PPLMetric(0, 0),
        ]
        actual_output = PPLMetric.from_mask(token_values, token_mask)

        self.assertEqual(len(actual_output), len(output))
        # Because Metric.from_mask() calls Metric.many(), which in turn converts tensors to lists,
        # it possible for the actual and expected outputs to be close to each other but not exactly equal.
        for a, o in zip(actual_output, output):
            self.assertIsInstance(a, type(o))
            self.assertAlmostEqual(a.value(), o.value(), places=6)


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
        assert m.report()['key'] == 6

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

    def test_recent(self):
        m = Metrics()
        m2 = Metrics(shared=m.share())
        m.add('test', SumMetric(1))
        assert m.report() == {'test': 1}
        assert m.report_recent() == {'test': 1}
        m.clear_recent()
        m.add('test', SumMetric(2))
        assert m.report() == {'test': 3}
        assert m.report_recent() == {'test': 2}
        assert m2.report() == {'test': 3}
        assert m2.report_recent() == {}
        m2.add('test', SumMetric(3))
        assert m2.report() == {'test': 6}
        assert m.report() == {'test': 6}
        assert m2.report_recent() == {'test': 3}
        assert m.report_recent() == {'test': 2}
        m2.clear_recent()
        assert m2.report() == {'test': 6}
        assert m.report() == {'test': 6}
        assert m2.report_recent() == {}
        assert m.report_recent() == {'test': 2}
        m.clear_recent()
        assert m2.report() == {'test': 6}
        assert m.report() == {'test': 6}
        assert m.report_recent() == {}


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

    def test_auc_metrics(self):
        class_name = 'class_notok'
        class_to_int = {'class_notok': 1, 'class_ok': 0}
        decimal_place = 3
        # task 1; borrowing example from scikit learn
        task1_probabilities = [0.1, 0.4, 0.35, 0.8]
        task1_gold_labels = ['class_ok', 'class_ok', 'class_notok', 'class_notok']
        task1_pos_buckets = {0.35: 1, 0.8: 1}
        task1_neg_buckets = {0.1: 1, 0.4: 1}
        task1_exp_fp_tp = {
            # thres: (False positives, True positives)
            0.1: (2, 2),
            0.35: (1, 2),
            0.4: (1, 1),
            0.8: (0, 1),
            '_': (0, 0),
        }

        # task 2; checking with an odd number
        task2_probabilities = [0.05, 0.2, 0.6]
        task2_gold_labels = ['class_ok', 'class_ok', 'class_notok']
        task2_pos_buckets = {0.6: 1}
        task2_neg_buckets = {0.05: 1, 0.2: 1}
        task2_exp_fp_tp = {0.05: (2, 1), 0.2: (1, 1), 0.6: (0, 1), 1.5: (0, 0)}

        # task 3: combining task 1 and task 2
        task3_probabilities = task1_probabilities + task2_probabilities
        task3_gold_labels = task1_gold_labels + task2_gold_labels
        task3_pos_buckets = {0.35: 1, 0.8: 1, 0.6: 1}
        task3_neg_buckets = {0.1: 1, 0.4: 1, 0.05: 1, 0.2: 1}
        task3_exp_fp_tp = {
            # threshold: FP, TP
            0.05: (4, 3),
            0.1: (3, 3),
            0.2: (2, 3),
            0.35: (1, 3),
            0.4: (1, 2),
            0.6: (0, 2),
            0.8: (0, 1),
            '_': (0, 0),
        }

        # task 4: testing when there's ones in the same bucket
        task4_probabilities = [0.1, 0.400001, 0.4, 0.359, 0.35, 0.900001, 0.9]
        task4_gold_labels = [
            'class_ok',
            'class_ok',
            'class_ok',
            'class_notok',
            'class_notok',
            'class_notok',
            'class_notok',
        ]
        task4_neg_buckets = {0.1: 1, 0.4: 2}
        task4_pos_buckets = {0.35: 1, 0.359: 1, 0.9: 2}
        task4_exp_fp_tp = {
            # thres: (False positives, True positives)
            0.1: (3, 4),
            0.35: (2, 4),
            0.359: (2, 3),
            0.4: (2, 2),
            0.9: (0, 2),
            '_': (0, 0),
        }

        # task 5: testing when there's more difference in the bucket (similar to task 4),
        # but testing to make sure the rounding/flooring is correct, and the edge cases 0.0, 1.0
        task5_probabilities = [0, 0.8, 0.4009, 0.400, 0.359, 0.35, 0.9999, 0.999, 1]
        # 4 okay, 5 not okay
        task5_gold_labels = [
            'class_ok',
            'class_ok',
            'class_ok',
            'class_ok',
            'class_notok',
            'class_notok',
            'class_notok',
            'class_notok',
            'class_notok',
        ]
        task5_neg_buckets = {0: 1, 0.8: 1, 0.4: 2}
        task5_pos_buckets = {0.35: 1, 0.359: 1, 0.999: 2, 1: 1}
        task5_exp_fp_tp = {
            # thres: (False positives, True positives)
            0: (4, 5),
            0.35: (3, 5),
            0.359: (3, 4),
            0.4: (3, 3),
            0.8: (1, 3),
            0.9: (0, 3),
            1.0: (0, 1),
            '_': (0, 0),
        }

        # task 6: combining task 4 + task 5 (combining with same keys)
        task6_probabilities = task4_probabilities + task5_probabilities
        task6_gold_labels = task4_gold_labels + task5_gold_labels
        task6_neg_buckets = {0: 1, 0.8: 1, 0.4: 4, 0.1: 1}
        task6_pos_buckets = {0.35: 2, 0.359: 2, 0.9: 2, 0.999: 2, 1: 1}
        task6_exp_fp_tp = {
            # threshold: FP, TP
            0: (7, 9),
            0.1: (6, 9),
            0.35: (5, 9),
            0.359: (5, 7),
            0.4: (5, 5),
            0.8: (1, 5),
            0.9: (0, 5),
            0.999: (0, 3),
            1: (0, 1),
            '_': (0, 0),
        }

        # run and check the TPs and FPs for singles
        task1_result = AUCMetrics.raw_data_to_auc(
            task1_gold_labels,
            task1_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task2_result = AUCMetrics.raw_data_to_auc(
            task2_gold_labels,
            task2_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task3_result = AUCMetrics.raw_data_to_auc(
            task3_gold_labels,
            task3_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task4_result = AUCMetrics.raw_data_to_auc(
            task4_gold_labels,
            task4_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task5_result = AUCMetrics.raw_data_to_auc(
            task5_gold_labels,
            task5_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task6_result = AUCMetrics.raw_data_to_auc(
            task6_gold_labels,
            task6_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )

        # check the buckets first
        self.assertEqual(task1_result._pos_dict, task1_pos_buckets)
        self.assertEqual(task1_result._neg_dict, task1_neg_buckets)
        self.assertEqual(task2_result._pos_dict, task2_pos_buckets)
        self.assertEqual(task2_result._neg_dict, task2_neg_buckets)
        self.assertEqual(task3_result._pos_dict, task3_pos_buckets)
        self.assertEqual(task3_result._neg_dict, task3_neg_buckets)
        self.assertEqual(task4_result._pos_dict, task4_pos_buckets)
        self.assertEqual(task4_result._neg_dict, task4_neg_buckets)
        self.assertEqual(task5_result._pos_dict, task5_pos_buckets)
        self.assertEqual(task5_result._neg_dict, task5_neg_buckets)
        self.assertEqual(task6_result._pos_dict, task6_pos_buckets)
        self.assertEqual(task6_result._neg_dict, task6_neg_buckets)

        # then check fp, tp
        self.assertEqual(set(task1_result._calc_fp_tp()), set(task1_exp_fp_tp.values()))
        self.assertEqual(set(task2_result._calc_fp_tp()), set(task2_exp_fp_tp.values()))
        self.assertEqual(set(task3_result._calc_fp_tp()), set(task3_exp_fp_tp.values()))
        self.assertEqual(set(task4_result._calc_fp_tp()), set(task4_exp_fp_tp.values()))
        self.assertEqual(set(task5_result._calc_fp_tp()), set(task5_exp_fp_tp.values()))
        self.assertEqual(set(task6_result._calc_fp_tp()), set(task6_exp_fp_tp.values()))

        # check that merging also produces the same results
        task3_result = task1_result + task2_result
        self.assertEqual(task3_result._pos_dict, task3_pos_buckets)
        self.assertEqual(task3_result._neg_dict, task3_neg_buckets)
        self.assertEqual(set(task3_result._calc_fp_tp()), set(task3_exp_fp_tp.values()))

        task6_result = task4_result + task5_result
        self.assertEqual(task6_result._pos_dict, task6_pos_buckets)
        self.assertEqual(task6_result._neg_dict, task6_neg_buckets)
        self.assertEqual(set(task6_result._calc_fp_tp()), set(task6_exp_fp_tp.values()))

        # now actually testing the area under curve
        from sklearn.metrics import roc_auc_score

        task1_labels_int = [
            class_to_int[gold_label] for gold_label in task1_gold_labels
        ]
        task2_labels_int = [
            class_to_int[gold_label] for gold_label in task2_gold_labels
        ]
        task3_labels_int = [
            class_to_int[gold_label] for gold_label in task3_gold_labels
        ]
        task4_labels_int = [
            class_to_int[gold_label] for gold_label in task4_gold_labels
        ]
        task5_labels_int = [
            class_to_int[gold_label] for gold_label in task5_gold_labels
        ]
        task6_labels_int = [
            class_to_int[gold_label] for gold_label in task6_gold_labels
        ]

        self.assertAlmostEqual(
            roc_auc_score(task1_labels_int, task1_probabilities), task1_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task2_labels_int, task2_probabilities), task2_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task3_labels_int, task3_probabilities), task3_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task4_labels_int, task4_probabilities), task4_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task5_labels_int, task5_probabilities), task5_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task6_labels_int, task6_probabilities), task6_result.value()
        )

        # last task: adding everything together; uses task 3 & 6
        # gonna just check roc scores
        task_all_gold_labels = task3_gold_labels + task6_gold_labels
        task_all_labels_int = task3_labels_int + task6_labels_int
        task_all_probabilities = task3_probabilities + task6_probabilities

        task_all_result = task3_result + task6_result

        self.assertAlmostEqual(
            roc_auc_score(task_all_labels_int, task_all_probabilities),
            task_all_result.value(),
        )

        task_all_result2 = AUCMetrics.raw_data_to_auc(
            task_all_gold_labels, task_all_probabilities, class_name
        )
        self.assertAlmostEqual(
            roc_auc_score(task_all_labels_int, task_all_probabilities),
            task_all_result2.value(),
        )

        ### now reusing the tests for the other class, just checking rocs
        ## for binary classes, they should be the same?
        class_name = 'class_ok'
        task1_probabilities = [1 - curr_prob for curr_prob in task1_probabilities]
        task2_probabilities = [1 - curr_prob for curr_prob in task2_probabilities]
        task3_probabilities = [1 - curr_prob for curr_prob in task3_probabilities]
        task4_probabilities = [1 - curr_prob for curr_prob in task4_probabilities]
        task5_probabilities = [1 - curr_prob for curr_prob in task5_probabilities]
        task6_probabilities = [1 - curr_prob for curr_prob in task6_probabilities]

        task1_labels_int = [1 - curr for curr in task1_labels_int]
        task2_labels_int = [1 - curr for curr in task2_labels_int]
        task3_labels_int = [1 - curr for curr in task3_labels_int]
        task4_labels_int = [1 - curr for curr in task4_labels_int]
        task5_labels_int = [1 - curr for curr in task5_labels_int]
        task6_labels_int = [1 - curr for curr in task6_labels_int]

        # get the results
        task1_result = AUCMetrics.raw_data_to_auc(
            task1_gold_labels,
            task1_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task2_result = AUCMetrics.raw_data_to_auc(
            task2_gold_labels,
            task2_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task3_result = AUCMetrics.raw_data_to_auc(
            task3_gold_labels,
            task3_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task4_result = AUCMetrics.raw_data_to_auc(
            task4_gold_labels,
            task4_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task5_result = AUCMetrics.raw_data_to_auc(
            task5_gold_labels,
            task5_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )
        task6_result = AUCMetrics.raw_data_to_auc(
            task6_gold_labels,
            task6_probabilities,
            class_name,
            max_bucket_dec_places=decimal_place,
        )

        # check against roc_auc_score
        self.assertAlmostEqual(
            roc_auc_score(task1_labels_int, task1_probabilities), task1_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task2_labels_int, task2_probabilities), task2_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task3_labels_int, task3_probabilities), task3_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task4_labels_int, task4_probabilities), task4_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task5_labels_int, task5_probabilities), task5_result.value()
        )
        self.assertAlmostEqual(
            roc_auc_score(task6_labels_int, task6_probabilities), task6_result.value()
        )

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


class TestDistinct(unittest.TestCase):
    def test_inter_distinct(self):
        # 3 n-grams, all appearing once
        m = InterDistinctMetric.compute("this is some test", 2)
        self.assertAlmostEqual(m, 1.0)
        # 3-grams, each appearing twice
        self.assertAlmostEqual(m + m, 0.5)

    def test_inter_distinct_unigram(self):
        m1 = InterDistinctMetric.compute("this test", 1)
        self.assertAlmostEqual(m1, 1.0, delta=0.001)
        m2 = InterDistinctMetric.compute("another test", 1)
        self.assertAlmostEqual(m2, 1.0, delta=0.001)
        # we now have 4 tokens, 3 words
        self.assertAlmostEqual(m1 + m2, 3 / 4)

    def test_intra_distinct(self):
        # 4/5 are unique
        m1 = IntraDistinctMetric.compute("this is some test test", 1)
        self.assertAlmostEqual(m1, 4 / 5)
        m2 = IntraDistinctMetric.compute("this test test test test", 1)
        self.assertAlmostEqual(m2, 2 / 5)
        self.assertAlmostEqual(m1 + m2, 3 / 5)


@testing_utils.skipUnlessFairseq
class TestFairseqBleuMetric(unittest.TestCase):
    """
    We're just going to compare that scores from Fairseq's Bleu scorer are the same as
    our scorer.
    """

    def test_scorer(self):
        import random

        vocab_length = num_ex = 100
        ex_length = 10
        pad_idx = 0
        eos_idx = 1
        unk_idx = 2

        try:
            from fairseq.scoring.bleu import Scorer
            from fairseq.scoring.bleu import BleuConfig

            fairseq_metrics: Scorer = Scorer(
                BleuConfig(pad=pad_idx, eos=eos_idx, unk=unk_idx)
            )
        except ImportError:
            # Bleuconfig is a recent version of fairseq
            fairseq_metrics: Scorer = Scorer(pad_idx, eos_idx, unk_idx)

        parlai_metrics: Dict[int, FairseqBleuMetric] = {k: [] for k in range(1, 5)}

        for _ in range(num_ex):
            guess = torch.LongTensor(random.sample(range(vocab_length), ex_length))
            answer = torch.LongTensor(random.sample(range(vocab_length), ex_length))

            parlai_bleu = FairseqBleuMetric.compute_many(
                guess, answer.unsqueeze(0), pad_idx, eos_idx, unk_idx
            )
            for i, bleu in enumerate(parlai_bleu):
                parlai_metrics[i + 1].append(bleu)
            fairseq_metrics.add(answer.int(), guess.int())

        parlai_bleus = {}
        for k, v in parlai_metrics.items():
            total = v[0]
            for vv in v[1:]:
                total = total + vv
            parlai_bleus[k] = total

        fairseq_bleus = {k: fairseq_metrics.score(order=k) for k in range(1, 5)}

        assert all(
            parlai_bleus[k] == fairseq_bleus[k] for k in range(1, 5)
        ), f'{parlai_bleus}\n{fairseq_bleus}'


if __name__ == '__main__':
    unittest.main()

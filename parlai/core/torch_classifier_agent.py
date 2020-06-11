#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Torch Classifier Agents classify text into a fixed set of labels.
"""


from parlai.core.opt import Opt
from parlai.utils.torch import PipelineHelper
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.core.metrics import Metric, AverageMetric
from typing import List, Optional, Tuple, Dict
from parlai.utils.typing import TScalar
import parlai.utils.logging as logging


import torch
import torch.nn.functional as F


class ConfusionMatrixMetric(Metric):
    """
    Class that keeps count of the confusion matrix for classification.

    Also provides helper methods computes precision, recall, f1, weighted_f1 for
    classification.
    """

    __slots__ = (
        '_true_positives',
        '_true_negatives',
        '_false_positives',
        '_false_negatives',
    )

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(
        self,
        true_positives: TScalar = 0,
        true_negatives: TScalar = 0,
        false_positives: TScalar = 0,
        false_negatives: TScalar = 0,
    ) -> None:
        self._true_positives = self.as_number(true_positives)
        self._true_negatives = self.as_number(true_negatives)
        self._false_positives = self.as_number(false_positives)
        self._false_negatives = self.as_number(false_negatives)

    def __add__(
        self, other: Optional['ConfusionMatrixMetric']
    ) -> 'ConfusionMatrixMetric':
        # NOTE: hinting can be cleaned up with "from __future__ import annotations" when
        # we drop Python 3.6
        if other is None:
            return self
        assert isinstance(other, ConfusionMatrixMetric)
        full_true_positives: TScalar = self._true_positives + other._true_positives
        full_true_negatives: TScalar = self._true_negatives + other._true_negatives
        full_false_positives: TScalar = self._false_positives + other._false_positives
        full_false_negatives: TScalar = self._false_negatives + other._false_negatives

        # always keep the same return type
        return type(self)(
            true_positives=full_true_positives,
            true_negatives=full_true_negatives,
            false_positives=full_false_positives,
            false_negatives=full_false_negatives,
        )

    @staticmethod
    def compute_many(
        true_positives: TScalar = 0,
        true_negatives: TScalar = 0,
        false_positives: TScalar = 0,
        false_negatives: TScalar = 0,
    ) -> Tuple['PrecisionMetric', 'RecallMetric', 'ClassificationF1Metric']:
        return (
            PrecisionMetric(
                true_positives, true_negatives, false_positives, false_negatives
            ),
            RecallMetric(
                true_positives, true_negatives, false_positives, false_negatives
            ),
            ClassificationF1Metric(
                true_positives, true_negatives, false_positives, false_negatives
            ),
        )

    @staticmethod
    def compute_metrics(
        predictions: List[str], gold_labels: List[str], positive_class: str
    ) -> Tuple[
        List['PrecisionMetric'], List['RecallMetric'], List['ClassificationF1Metric']
    ]:
        precisions = []
        recalls = []
        f1s = []
        for predicted, gold_label in zip(predictions, gold_labels):
            true_positives = int(
                predicted == positive_class and gold_label == positive_class
            )
            true_negatives = int(
                predicted != positive_class and gold_label != positive_class
            )
            false_positives = int(
                predicted == positive_class and gold_label != positive_class
            )
            false_negatives = int(
                predicted != positive_class and gold_label == positive_class
            )
            precision, recall, f1 = ConfusionMatrixMetric.compute_many(
                true_positives, true_negatives, false_positives, false_negatives
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return precisions, recalls, f1s


class PrecisionMetric(ConfusionMatrixMetric):
    """
    Class that takes in a ConfusionMatrixMetric and computes precision for classifier.
    """

    def value(self) -> float:
        if self._true_positives == 0:
            return 0.0
        else:
            return self._true_positives / (self._true_positives + self._false_positives)


class RecallMetric(ConfusionMatrixMetric):
    """
    Class that takes in a ConfusionMatrixMetric and computes recall for classifier.
    """

    def value(self) -> float:
        if self._true_positives == 0:
            return 0.0
        else:
            return self._true_positives / (self._true_positives + self._false_negatives)


class ClassificationF1Metric(ConfusionMatrixMetric):
    """
    Class that takes in a ConfusionMatrixMetric and computes f1 for classifier.
    """

    def value(self) -> float:
        if self._true_positives == 0:
            return 0.0
        else:
            numer = 2 * self._true_positives
            denom = numer + self._false_negatives + self._false_positives
            return numer / denom


class WeightedF1Metric(Metric):
    """
    Class that represents the weighted f1 from ClassificationF1Metric.
    """

    __slots__ = '_values'

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return True

    def __init__(self, metrics: Dict[str, ClassificationF1Metric]) -> None:
        self._values: Dict[str, ClassificationF1Metric] = metrics

    def __add__(self, other: Optional['WeightedF1Metric']) -> 'WeightedF1Metric':
        if other is None:
            return self
        assert isinstance(other, WeightedF1Metric)
        output: Dict[str, ClassificationF1Metric] = dict(**self._values)
        for k, v in other._values.items():
            output[k] = output.get(k, None) + v  # type: ignore
        return WeightedF1Metric(output)

    def value(self) -> float:
        weighted_f1 = 0.0
        values = list(self._values.values())
        if len(values) == 0:
            return weighted_f1
        total_examples = (
            values[0]._true_positives
            + values[0]._true_negatives
            + values[0]._false_positives
            + values[0]._false_negatives
        )
        for each in values:
            actual_positive = each._true_positives + each._false_negatives
            weighted_f1 += each.value() * (actual_positive / total_examples)
        return weighted_f1

    @staticmethod
    def compute_many(
        metrics: Dict[str, List[ClassificationF1Metric]]
    ) -> List['WeightedF1Metric']:
        weighted_f1s = [dict(zip(metrics, t)) for t in zip(*metrics.values())]
        return [WeightedF1Metric(metrics) for metrics in weighted_f1s]


class TorchClassifierAgent(TorchAgent):
    """
    Abstract Classifier agent. Only meant to be extended.

    TorchClassifierAgent aims to handle much of the bookkeeping any classification
    model.
    """

    @staticmethod
    def add_cmdline_args(parser):
        """
        Add CLI args.
        """
        TorchAgent.add_cmdline_args(parser)
        parser = parser.add_argument_group('Torch Classifier Arguments')
        # class arguments
        parser.add_argument(
            '--classes',
            type=str,
            nargs='*',
            default=None,
            help='the name of the classes.',
        )
        parser.add_argument(
            '--class-weights',
            type=float,
            nargs='*',
            default=None,
            help='weight of each of the classes for the softmax',
        )
        parser.add_argument(
            '--ref-class',
            type=str,
            default=None,
            hidden=True,
            help='the class that will be used to compute '
            'precision and recall. By default the first '
            'class.',
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.5,
            help='during evaluation, threshold for choosing '
            'ref class; only applies to binary '
            'classification',
        )
        # interactive mode
        parser.add_argument(
            '--print-scores',
            type='bool',
            default=False,
            help='print probability of chosen class during ' 'interactive mode',
        )
        # miscellaneous arguments
        parser.add_argument(
            '--data-parallel',
            type='bool',
            default=False,
            help='uses nn.DataParallel for multi GPU',
        )
        parser.add_argument(
            '--classes-from-file',
            type=str,
            default=None,
            help='loads the list of classes from a file',
        )
        parser.add_argument(
            '--ignore-labels',
            type='bool',
            default=None,
            help='Ignore labels provided to model',
        )

    def __init__(self, opt: Opt, shared=None):
        init_model, self.is_finetune = self._get_init_model(opt, shared)
        super().__init__(opt, shared)

        # set up classes
        if opt.get('classes') is None and opt.get('classes_from_file') is None:
            raise RuntimeError(
                'Must specify --classes or --classes-from-file argument.'
            )
        if not shared:
            if opt['classes_from_file'] is not None:
                with open(opt['classes_from_file']) as f:
                    self.class_list = f.read().splitlines()
            else:
                self.class_list = opt['classes']
            self.class_dict = {val: i for i, val in enumerate(self.class_list)}
            if opt.get('class_weights', None) is not None:
                self.class_weights = opt['class_weights']
            else:
                self.class_weights = [1.0 for c in self.class_list]
            self.reset_metrics()
        else:
            self.class_list = shared['class_list']
            self.class_dict = shared['class_dict']
            self.class_weights = shared['class_weights']

        # in binary classfication, opt['threshold'] applies to ref class
        if opt['ref_class'] is None or opt['ref_class'] not in self.class_dict:
            self.ref_class = self.class_list[0]
        else:
            self.ref_class = opt['ref_class']
            ref_class_id = self.class_list.index(self.ref_class)
            if ref_class_id != 0:
                # move to the front of the class list
                self.class_list.insert(0, self.class_list.pop(ref_class_id))

        # set up threshold, only used in binary classification
        if len(self.class_list) == 2 and opt.get('threshold', 0.5) != 0.5:
            self.threshold = opt['threshold']
        else:
            self.threshold = None

        # set up model and optimizers

        if shared:
            self.model = shared['model']
        else:
            self.model = self.build_model()
            self.criterion = self.build_criterion()
            if self.model is None or self.criterion is None:
                raise AttributeError(
                    'build_model() and build_criterion() need to return the model or criterion'
                )
            if init_model:
                logging.info(f'Loading existing model parameters from {init_model}')
                self.load(init_model)
            if self.use_cuda:
                if self.model_parallel:
                    self.model = PipelineHelper().make_parallel(self.model)
                else:
                    self.model.cuda()
                if self.data_parallel:
                    self.model = torch.nn.DataParallel(self.model)
                self.criterion.cuda()

        if shared:
            # We don't use get here because hasattr is used on optimizer later.
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        elif self._should_initialize_optimizer():
            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            self.init_optim(optim_params)
            self.build_lr_scheduler()

    def build_criterion(self):
        weight_tensor = torch.FloatTensor(self.class_weights)
        return torch.nn.CrossEntropyLoss(weight=weight_tensor, reduction='none')

    def share(self):
        """
        Share model parameters.
        """
        shared = super().share()
        shared['class_dict'] = self.class_dict
        shared['class_list'] = self.class_list
        shared['class_weights'] = self.class_weights
        shared['model'] = self.model
        if hasattr(self, 'optimizer'):
            shared['optimizer'] = self.optimizer
        return shared

    def _get_labels(self, batch):
        """
        Obtain the correct labels.

        Raises a ``KeyError`` if one of the labels is not in the class list.
        """
        try:
            labels_indices_list = [self.class_dict[label] for label in batch.labels]
        except KeyError as e:
            warn_once('One of your labels is not in the class list.')
            raise e

        labels_tensor = torch.LongTensor(labels_indices_list)
        if self.use_cuda:
            labels_tensor = labels_tensor.cuda()
        return labels_tensor

    def _update_confusion_matrix(self, batch, predictions):
        """
        Update the confusion matrix given the batch and predictions.

        :param predictions:
            (list of string of length batchsize) label predicted by the
            classifier
        :param batch:
            a Batch object (defined in torch_agent.py)
        """
        f1_dict = {}
        for class_name in self.class_list:
            prec_str = f'class_{class_name}_prec'
            recall_str = f'class_{class_name}_recall'
            f1_str = f'class_{class_name}_f1'
            precision, recall, f1 = ConfusionMatrixMetric.compute_metrics(
                predictions, batch.labels, class_name
            )
            f1_dict[class_name] = f1
            self.record_local_metric(prec_str, precision)
            self.record_local_metric(recall_str, recall)
            self.record_local_metric(f1_str, f1)
        self.record_local_metric('weighted_f1', WeightedF1Metric.compute_many(f1_dict))

    def _format_interactive_output(self, probs, prediction_id):
        """
        Format interactive mode output with scores.
        """
        preds = []
        for i, pred_id in enumerate(prediction_id.tolist()):
            prob = round_sigfigs(probs[i][pred_id], 4)
            preds.append(
                'Predicted class: {}\nwith probability: {}'.format(
                    self.class_list[pred_id], prob
                )
            )
        return preds

    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        if batch.text_vec is None:
            return Output()
        self.model.train()
        self.optimizer.zero_grad()

        # calculate loss
        labels = self._get_labels(batch)
        scores = self.score(batch)
        loss = self.criterion(scores, labels)
        self.record_local_metric('loss', AverageMetric.many(loss))
        loss = loss.mean()
        loss.backward()
        self.update_params()

        # get predictions
        _, prediction_id = torch.max(scores.cpu(), 1)
        preds = [self.class_list[idx] for idx in prediction_id]
        self._update_confusion_matrix(batch, preds)

        return Output(preds)

    def eval_step(self, batch):
        """
        Train on a single batch of examples.
        """
        if batch.text_vec is None:
            return

        self.model.eval()
        scores = self.score(batch)
        probs = F.softmax(scores, dim=1)
        if self.threshold is None:
            _, prediction_id = torch.max(probs.cpu(), 1)
        else:
            ref_prob = probs.cpu()[:, 0]
            # choose ref class if Prob(ref class) > threshold
            prediction_id = (ref_prob <= self.threshold).to(torch.int64)
        preds = [self.class_list[idx] for idx in prediction_id]

        if batch.labels is None or self.opt['ignore_labels']:
            # interactive mode
            if self.opt.get('print_scores', False):
                preds = self._format_interactive_output(probs, prediction_id)
        else:
            labels = self._get_labels(batch)
            loss = self.criterion(scores, labels)
            self.record_local_metric('loss', AverageMetric.many(loss))
            loss = loss.mean()
            self._update_confusion_matrix(batch, preds)

        if self.opt.get('print_scores', False):
            return Output(preds, probs=probs.cpu())
        else:
            return Output(preds)

    def score(self, batch):
        """
        Given a batch and labels, returns the scores.

        :param batch:
            a Batch object (defined in torch_agent.py)
        :return:
            a [bsz, num_classes] FloatTensor containing the score of each
            class.
        """
        raise NotImplementedError('Abstract class: user must implement score()')

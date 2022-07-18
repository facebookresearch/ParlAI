#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Torch Classifier Agents classify text into a fixed set of labels.
"""


from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.utils.torch import PipelineHelper, total_parameters, trainable_parameters
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.core.metrics import Metric, AverageMetric
from typing import List, Optional, Tuple, Dict, Union
from typing import Counter
from parlai.utils.typing import TScalar
from parlai.utils.io import PathManager
from sklearn.metrics import auc

import parlai.utils.logging as logging
import torch.nn.functional as F
import torch
import math


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


class AUCMetrics(Metric):
    """
    Computes Area Under ROC Curve (AUC) metrics.

    Does so by keeping track of positives' and negatives' probability score counts in
    Counters or dictionaries. Note the introduction of `max_bucket_dec_places`; this
    integer number determines the number of digits to save for the probability scores. A
    higher `max_bucket_dec_places` will a more accurate estimate of the exact AUC
    metric, but may also use more space.

    NOTE: currently only used for classifiers in the `eval_model` script; to use,
    add the argument `-auc <max_bucket_dec_places>` when calling `eval_model` script
    """

    @property
    def macro_average(self) -> bool:
        """
        Indicates whether this metric should be macro-averaged when globally reported.
        """
        return False

    @classmethod
    def raw_data_to_auc(
        cls,
        true_labels: List[Union[int, str]],
        pos_probs: List[float],
        class_name,
        max_bucket_dec_places: int = 3,
    ):
        auc_object = cls(class_name, max_bucket_dec_places=max_bucket_dec_places)
        auc_object.update_raw(
            true_labels=true_labels, pos_probs=pos_probs, class_name=class_name
        )
        return auc_object

    def __init__(
        self,
        class_name: Union[int, str],
        max_bucket_dec_places: int = 3,
        pos_dict: Optional[Counter[float]] = None,
        neg_dict: Optional[Counter[float]] = None,
    ):
        # `_pos_dict` keeps track of the probabilities of the positive class
        self._pos_dict = pos_dict if pos_dict else Counter()
        # `_neg_dict` keeps track of the probabilities of the negative class
        self._neg_dict = neg_dict if neg_dict else Counter()
        self._class_name = class_name
        self._max_bucket_dec_places = max_bucket_dec_places

    def update_raw(
        self, true_labels: List[Union[int, str]], pos_probs: List[float], class_name
    ):
        """
        given the true/golden labels and the probabilities of the positive class, we
        will update our bucket dictionaries of positive and negatives (based on the
        class_name); `max_bucket_dec_places` is also used here to round the
        probabilities and possibly.
        """
        assert self._class_name == class_name
        assert len(true_labels) == len(pos_probs)

        TO_INT_FACTOR = 10**self._max_bucket_dec_places
        # add the upper and lower bound of the values
        for label, prob in zip(true_labels, pos_probs):
            # calculate the upper and lower bound of the values
            prob_down = math.floor(prob * TO_INT_FACTOR) / TO_INT_FACTOR
            if label == self._class_name:
                interested_dict = self._pos_dict
            else:
                interested_dict = self._neg_dict
            if interested_dict.get(prob_down):
                interested_dict[prob_down] += 1
            else:
                interested_dict[prob_down] = 1

    def __add__(self, other: Optional['AUCMetrics']) -> 'AUCMetrics':
        if other is None:
            return self
        assert isinstance(other, AUCMetrics)
        assert other._class_name == self._class_name
        all_pos_dict = self._pos_dict + other._pos_dict
        all_neg_dict = self._neg_dict + other._neg_dict

        return AUCMetrics(
            self._class_name, pos_dict=all_pos_dict, neg_dict=all_neg_dict
        )

    def _calc_fp_tp(self) -> List[Tuple[int]]:
        """
        Calculates the False Positives and True positives; returned as a list of pairs:

        `[(fp, tp)]`
        """
        all_thresholds = sorted(
            set(list(self._pos_dict.keys()) + list(self._neg_dict.keys()))
        )
        # sorted in ascending order,
        # so adding a upper bound so that its tp, fp is (0, 0)
        all_thresholds.append(all_thresholds[-1] + 1)
        L = len(all_thresholds)
        # false positives, true positives
        fp_tp = [(0, 0)]

        # the biggest one is always (0,0), so skip that one
        for i in range(L - 2, -1, -1):
            fp, tp = fp_tp[-1]
            thres = all_thresholds[i]
            # false positives
            fp += self._neg_dict.get(thres, 0)
            # true positives
            tp += self._pos_dict.get(thres, 0)
            fp_tp.append((fp, tp))
        return fp_tp

    def _calc_fpr_tpr(self) -> Tuple[Union[List[int], int]]:
        """
        Calculates the false positive rates and true positive rates Also returns the
        total number of positives and negatives; returned as a list of pairs and two
        integers:

        `([(fpr, tpr)], positives, negatives)`; note that if the total
        negatives/positives is 0, then will return 0 for either fpr/tpr instead of
        raising an error
        """
        _tot_pos = sum(self._pos_dict.values())
        _tot_neg = sum(self._neg_dict.values())
        fp_tp = self._calc_fp_tp()
        fps, tps = list(zip(*fp_tp))
        if _tot_neg == 0:
            fpr = [0] * len(fps)
        else:
            fpr = [fp / _tot_neg for fp in fps]

        if _tot_pos == 0:
            tpr = [0] * len(tps)
        else:
            tpr = [tp / _tot_pos for tp in tps]

        return (list(zip(fpr, tpr)), _tot_pos, _tot_neg)

    def value(self) -> float:
        fpr_tpr, _tot_pos, _tot_neg = self._calc_fpr_tpr()

        if _tot_pos == 0 and _tot_neg == 0:
            return 0

        # auc needs x-axis to be sorted
        fpr_tpr.sort()
        fpr, tpr = list(zip(*fpr_tpr))
        return auc(fpr, tpr)


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
        total_examples = sum(
            [each._true_positives + each._false_negatives for each in values]
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

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add CLI args.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
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
        parser.add_argument(
            '--update-classifier-head-only',
            type='bool',
            default=False,
            help='Freeze the encoder and update the classifier head only',
        )
        parser.set_defaults(use_reply='none')
        return parser

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
                with PathManager.open(opt['classes_from_file']) as f:
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

        # set up calculating auc
        self.calc_auc = opt.get('area_under_curve_digits', -1) > 0

        if self.calc_auc:
            self.auc_bucket_decimal_size = opt.get('area_under_curve_digits')
            if opt.get('area_under_curve_class') is None:
                # self.auc_class_ind
                interested_classes = self.class_list
            else:
                interested_classes = opt.get('area_under_curve_class')
            try:
                self.auc_class_indices = [
                    self.class_dict[class_name] for class_name in interested_classes
                ]
            except Exception:
                raise RuntimeError(
                    f'The inputted classes for auc were probably invalid.\n Current class names: {self.class_list} \n Names of AUC classes passed in: {interested_classes}'
                )
            self.reset_auc()

        # set up model and optimizers
        states = {}
        if shared:
            self.model = shared['model']
        else:
            self.model = self.build_model()
            # freeze the encoder and update the classifier only
            if opt.get("update_classifier_head_only", False):
                for _param_name, _param_value in self.model.named_parameters():
                    if not _param_name.startswith('additional_linear_layer'):
                        _param_value.requires_grad = False

            self.criterion = self.build_criterion()
            if self.model is None or self.criterion is None:
                raise AttributeError(
                    'build_model() and build_criterion() need to return the model or criterion'
                )
            if init_model:
                logging.info(f'Loading existing model parameters from {init_model}')
                states = self.load(init_model)
            if self.use_cuda:
                if self.model_parallel:
                    ph = PipelineHelper()
                    ph.check_compatibility(self.opt)
                    self.model = ph.make_parallel(self.model)
                else:
                    self.model.cuda()
                if self.data_parallel:
                    self.model = torch.nn.DataParallel(self.model)
                self.criterion.cuda()

            train_params = trainable_parameters(self.model)
            total_params = total_parameters(self.model)
            logging.info(
                f"Total parameters: {total_params:,d} ({train_params:,d} trainable)"
            )

        if shared:
            # We don't use get here because hasattr is used on optimizer later.
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        elif self._should_initialize_optimizer():
            was_reset = self.init_optim(
                [p for p in self.model.parameters() if p.requires_grad],
                optim_states=states.get('optimizer'),
                saved_optim_type=states.get('optimizer_type'),
                is_finetune=self.is_finetune,
            )
            if was_reset:
                logging.warning("Optimizer was reset. Also resetting LR scheduler.")
            self.build_lr_scheduler(states, hard_reset=self.is_finetune or was_reset)

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

    def _update_aucs(self, batch, probs):
        probs_arr = probs.detach().cpu().numpy()
        for index, curr_auc in zip(self.auc_class_indices, self.aucs):
            class_probs = probs_arr[:, index]
            curr_auc.update_raw(batch.labels, class_probs, self.class_list[index])

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
        self.backward(loss)
        self.update_params()

        # get predictions
        _, prediction_id = torch.max(scores.cpu(), 1)
        preds = [self.class_list[idx] for idx in prediction_id]
        self._update_confusion_matrix(batch, preds)

        return Output(preds)

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None:
            return

        self.model.eval()
        scores = self.score(batch)
        probs = F.softmax(scores, dim=1)

        if self.calc_auc:
            self._update_aucs(batch, probs)

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
            return Output(preds, class_list=[self.class_list], probs=probs.tolist())
        if self.opt.get('return_cand_scores', False):
            sorted_scores, ranks = probs.sort(1, descending=True)
            sorted_scores = sorted_scores.cpu()
            text_cands = []
            for i in range(0, ranks.size(0)):
                ordered_list = [self.class_list[i] for i in ranks[i]]
                text_cands.append(ordered_list)
            return Output(
                preds, text_candidates=text_cands, sorted_scores=sorted_scores
            )
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

    def reset_auc(self):
        if self.calc_auc:
            self.aucs = [
                AUCMetrics(
                    class_name=self.class_list[index],
                    max_bucket_dec_places=self.auc_bucket_decimal_size,
                )
                for index in self.auc_class_indices
            ]

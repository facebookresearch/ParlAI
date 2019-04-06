#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.distributed_utils import is_distributed
from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.utils import round_sigfigs, warn_once
from collections import defaultdict

import torch
import torch.nn.functional as F


class TorchClassifierAgent(TorchAgent):
    """
    Abstract Classifier agent. Only meant to be extended.

    TorchClassifierAgent aims to handle much of the bookkeeping any
    classification model.
    """

    @staticmethod
    def add_cmdline_args(parser):
        TorchAgent.add_cmdline_args(parser)
        parser = parser.add_argument_group('Torch Classifier Arguments')
        # class arguments
        parser.add_argument('--classes', type=str, nargs='*', default=None,
                            help='the name of the classes.')
        parser.add_argument('--class-weights', type=float, nargs='*', default=None,
                            help='weight of each of the classes for the softmax')
        parser.add_argument('--ref-class', type=str, default=None, hidden=True,
                            help='the class that will be used to compute '
                                 'precision and recall. By default the first '
                                 'class.')
        parser.add_argument('--threshold', type=float, default=0.5,
                            help='during evaluation, threshold for choosing '
                                 'ref class; only applies to binary '
                                 'classification')
        # interactive mode
        parser.add_argument('--print-scores', type='bool', default=False,
                            help='print probability of chosen class during '
                                 'interactive mode')
        # miscellaneous arguments
        parser.add_argument('--data-parallel', type='bool', default=False,
                            help='uses nn.DataParallel for multi GPU')
        parser.add_argument('--get-all-metrics', type='bool', default=True,
                            help='give prec/recall metrics for all classes')

    def __init__(self, opt, shared=None):
        init_model, _ = self._get_init_model(opt, shared)
        super().__init__(opt, shared)

        # set up classes
        if opt.get('classes') is None:
            raise RuntimeError(
                'Must specify --classes argument.'
            )
        if not shared:
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

        # get reference class; if opt['get_all_metrics'] is False, this is
        # used to compute metrics
        # in binary classfication, opt['threshold'] applies to ref class
        if (opt['ref_class'] is None or opt['ref_class'] not in
                self.class_dict):
            self.ref_class = self.class_list[0]
        else:
            self.ref_class = opt['ref_class']
            ref_class_id = self.class_list.index(self.ref_class)
            if ref_class_id != 0:
                # move to the front of the class list
                self.class_list.insert(0, self.class_list.pop(ref_class_id))
        if not opt['get_all_metrics']:
            warn_once('Using %s as the class for computing P, R, and F1' %
                      self.ref_class)

        # set up threshold, only used in binary classification
        if len(self.class_list) == 2 and opt.get('threshold', 0.5) != 0.5:
            self.threshold = opt['threshold']
        else:
            self.threshold = None

        # set up model and optimizers
        weight_tensor = torch.FloatTensor(self.class_weights)
        self.classifier_loss = torch.nn.CrossEntropyLoss(weight_tensor)
        if shared:
            self.model = shared['model']
        else:
            self.build_model()
            if init_model:
                print('Loading existing model parameters from ' + init_model)
                self.load(init_model)
        if self.use_cuda:
            self.model.cuda()
            self.classifier_loss.cuda()
            if self.opt['data_parallel']:
                if is_distributed():
                    raise ValueError(
                        'Cannot combine --data-parallel and distributed mode'
                    )
                self.model = torch.nn.DataParallel(self.model)
        if shared:
            # We don't use get here because hasattr is used on optimizer later.
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        else:
            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            self.init_optim(optim_params)
            self.build_lr_scheduler()

    def share(self):
        """Share model parameters."""
        shared = super().share()
        shared['class_dict'] = self.class_dict
        shared['class_list'] = self.class_list
        shared['class_weights'] = self.class_weights
        shared['metrics'] = self.metrics
        shared['model'] = self.model
        shared['optimizer'] = self.optimizer
        return shared

    def _get_labels(self, batch):
        """
        Obtain the correct labels. Raise an exception if one of the labels
        is not in the class list.
        """
        try:
            labels_indices_list = [self.class_dict[label] for label in
                                   batch.labels]
        except KeyError as e:
            print('One of your labels is not in the class list.')
            raise e
        labels_tensor = torch.LongTensor(labels_indices_list)
        if self.use_cuda:
            labels_tensor = labels_tensor.cuda()
        return labels_tensor

    def _update_confusion_matrix(self, batch, predictions):
        """
        Update the confusion matrix given the batch and predictions.

        :param batch:
            a Batch object (defined in torch_agent.py)
        :param predictions:
            (list of string of length batchsize) label predicted by the
            classifier
        """
        for i, pred in enumerate(predictions):
            label = batch.labels[i]
            self.metrics['confusion_matrix'][(label, pred)] += 1

    def _format_interactive_output(self, probs, prediction_id):
        """
        Nicely format interactive mode output when we want to also
        print the scores associated with the predictions.
        """
        preds = []
        for i, pred_id in enumerate(prediction_id.tolist()):
            prob = round_sigfigs(probs[i][pred_id], 4)
            preds.append('Predicted class: {}\nwith probability: {}'.format(
                         self.class_list[pred_id], prob))
        return preds

    def train_step(self, batch):
        """Train on a single batch of examples."""
        if batch.text_vec is None:
            return Output()
        self.model.train()
        self.optimizer.zero_grad()

        # calculate loss
        labels = self._get_labels(batch)
        scores = self.score(batch)
        loss = self.classifier_loss(scores, labels)
        loss.backward()
        self.update_params()

        # update metrics
        self.metrics['loss'] += loss.item()
        self.metrics['examples'] += len(batch.text_vec)

        # get predictions
        _, prediction_id = torch.max(scores.cpu(), 1)
        preds = [self.class_list[idx] for idx in prediction_id]
        self._update_confusion_matrix(batch, preds)

        return Output(preds)

    def eval_step(self, batch):
        """Train on a single batch of examples."""
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
            prediction_id = ref_prob <= self.threshold
        preds = [self.class_list[idx] for idx in prediction_id]

        if batch.labels is None:
            # interactive mode
            if self.opt.get('print_scores', False):
                preds = self._format_interactive_output(probs, prediction_id)
        else:
            labels = self._get_labels(batch)
            loss = self.classifier_loss(scores, labels)
            self.metrics['loss'] += loss.item()
            self.metrics['examples'] += len(batch.text_vec)
            self._update_confusion_matrix(batch, preds)

        return Output(preds)

    def reset_metrics(self):
        """Reset metrics."""
        self.metrics = {}
        super().reset_metrics()
        self.metrics['confusion_matrix'] = defaultdict(int)
        self.metrics['examples'] = 0
        self.metrics['loss'] = 0.0

    def _report_prec_recall_metrics(self, confmat, class_name, metrics):
        """
        Uses the confusion matrix to predict the recall and precision for
        class `class_name`. Returns the number of examples of each class.
        """
        eps = 0.00001  # prevent divide by zero errors
        true_positives = confmat[(class_name, class_name)]
        num_actual_positives = sum([confmat[(class_name, c)]
                                   for c in self.class_list]) + eps
        num_predicted_positives = sum([confmat[(c, class_name)]
                                      for c in self.class_list]) + eps

        recall_str = 'class_{}_recall'.format(class_name)
        prec_str = 'class_{}_prec'.format(class_name)
        f1_str = 'class_{}_f1'.format(class_name)

        # update metrics dict
        metrics[recall_str] = true_positives / num_actual_positives
        metrics[prec_str] = true_positives / num_predicted_positives
        metrics[f1_str] = 2 * ((metrics[recall_str] * metrics[prec_str]) /
                               (metrics[recall_str] + metrics[prec_str] + eps))

        return num_actual_positives

    def report(self):
        """Report loss as well as precision, recall, and F1 metrics."""
        m = super().report()
        examples = self.metrics['examples']
        if examples > 0:
            m['examples'] = examples
            m['mean_loss'] = self.metrics['loss'] / examples

            # get prec/recall metrics
            confmat = self.metrics['confusion_matrix']
            if self.opt.get('get_all_metrics'):
                metrics_list = self.class_list
            else:
                # only give prec/recall metrics for ref class
                metrics_list = [self.ref_class]

            examples_per_class = []
            for class_i in metrics_list:
                class_total = self._report_prec_recall_metrics(confmat,
                                                               class_i, m)
                examples_per_class.append(class_total)

            if len(examples_per_class) > 1:
                # get weighted f1
                f1 = 0
                total_exs = sum(examples_per_class)
                for i in range(len(self.class_list)):
                    f1 += ((examples_per_class[i] / total_exs) *
                           m['class_{}_f1'.format(self.class_list[i])])
                m['weighted_f1'] = f1

        for k, v in m.items():
            m[k] = round_sigfigs(v, 4)

        return m

    def score(self, batch):
        """
        Given a batch and labels, returns the scores.

        :param batch:
            a Batch object (defined in torch_agent.py)
        :return:
            a [bsz, num_classes] FloatTensor containing the score of each
            class.
        """
        raise NotImplementedError(
            'Abstract class: user must implement score()')

    def build_model(self):
        """Build a new model (implemented by children classes)."""
        raise NotImplementedError(
            'Abstract class: user must implement build_model()')

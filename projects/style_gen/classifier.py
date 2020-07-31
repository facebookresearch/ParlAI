#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent for classifying style with utterance(s) as context.
"""

from itertools import chain

import torch
from torch.nn import functional as F

from parlai.agents.transformer.transformer import (
    TransformerClassifierAgent,
    TransformerGeneratorAgent,
)
from parlai.core.metrics import AverageMetric
from parlai.core.torch_agent import Output
from parlai.core.torch_classifier_agent import ClassificationMixin
from parlai.utils.fp16 import FP16SafeCrossEntropy
from parlai.utils.misc import warn_once, round_sigfigs
from projects.style_gen.modules import (
    BatchWithPersonalities,
    ClassifierOnGeneratorModel,
)


class ClassifierAgent(ClassificationMixin, TransformerGeneratorAgent):
    """
    Agent that uses a generator model with a classifier head.

    Useful for performing classification with a pretrained generator model. The
    generator encoder/decoder weights can be frozen during classifier training.
    """

    # TODO: perhaps reduce the amount of code duplicated from TorchClassifierAgent. This
    #  would require modularizing several snippets of code inside TCA methods.

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add CLI args.
        """
        TransformerClassifierAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('ClassifierOnGenerator Arguments')
        agent.add_argument(
            '--freeze-enc-dec-weights',
            type='bool',
            default=False,
            help='Only train the classifier head and not the encoder and decoder',
        )
        agent.add_argument(
            '--personality-as-label',
            type='bool',
            default=True,
            help='The personality is in the label field instead of the personality field',
        )
        return agent

    def __init__(self, opt, shared=None):
        """
        Set up model.
        """

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
                self.class_weights = [1.0 for _ in self.class_list]
        else:
            self.class_list = shared['class_list']
            self.class_dict = shared['class_dict']
            self.class_weights = shared['class_weights']

        self.personality_as_label = opt['personality_as_label']

        super().__init__(opt, shared)

        # Override the criterion
        if not shared:
            self.criterion = self.build_criterion()
            if self.use_cuda:
                self.criterion.cuda()

        # Freeze generator encoder/decoder weights
        if opt['freeze_enc_dec_weights']:
            for param in chain(
                self.model.encoder.parameters(), self.model.decoder.parameters()
            ):
                param.requires_grad = False

    def build_model(self, states=None):
        """
        Build and return model.
        """
        model = ClassifierOnGeneratorModel(
            self.opt,
            self.dict,
            num_classes=len(self.class_list),
            personality_as_label=self.personality_as_label,
        )
        return model

    def build_criterion(self):
        weight_tensor = torch.FloatTensor(self.class_weights)
        if not self.fp16:
            return torch.nn.CrossEntropyLoss(weight=weight_tensor, reduction='none')
        else:
            # FP16 safe cross entropy (softmax done in FP32)
            return FP16SafeCrossEntropy(weight=weight_tensor, reduction='none')

    def load_state_dict(self, state_dict):
        """
        Override to add in the classifier head if it doesn't exist.
        """
        for tensor in ['weight', 'bias']:
            key = f'classifier_head.{tensor}'
            if key not in state_dict:
                state_dict[key] = getattr(self.model.classifier_head, tensor)
        super().load_state_dict(state_dict)

    def share(self):
        """
        Share model parameters.
        """
        shared = super().share()
        shared['class_dict'] = self.class_dict
        shared['class_list'] = self.class_list
        shared['class_weights'] = self.class_weights
        return shared

    def batchify(self, obs_batch, sort=False):
        base_batch = super().batchify(obs_batch, sort)
        if self.personality_as_label:
            return base_batch
        else:
            assert sort is False
            # Sorting would make it hard to line up the observations within one batch
            personalities = [
                obs['personality'] for obs in obs_batch if self.is_valid(obs)
            ]
            assert len(personalities) == len(base_batch.text_vec)
            batch_with_personalities = BatchWithPersonalities(
                personalities=personalities, **base_batch.__dict__
            )
            return batch_with_personalities

    def _get_label_tensor(self, batch):
        """
        Obtain the correct class labels.

        Raises a `KeyError` if one of the labels is not in the class list.
        """
        if self.personality_as_label:
            labels = batch.labels
        else:
            labels = batch.personalities
        try:
            labels_indices_list = [self.class_dict[label] for label in labels]
        except KeyError as e:
            warn_once('One of your labels is not in the class list.')
            raise e

        labels_tensor = torch.LongTensor(labels_indices_list)
        if self.use_cuda:
            labels_tensor = labels_tensor.cuda()
        return labels_tensor

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

    def batch_act(self, observations):
        """
        Overwriting ClassificationMixin.batch_act() in the case where the labels are
        stored in the "personality" field.
        """

        if self.personality_as_label:

            batch_reply = super().batch_act(observations)
            return batch_reply

        else:

            batch_reply = super(ClassificationMixin, self).batch_act(observations)

            preds = self._get_preds(batch_reply)
            if preds is None:
                return batch_reply

            labels = 'personality'
            labels_lst = [[label] for label in self._get_labels(observations, labels)]
            # The label is expected to be in a list like in the "labels" or
            # "eval_labels" fields
            self._update_confusion_matrix(preds, labels_lst)

            return batch_reply

    def train_step(self, batch):
        """
        Train on a single batch of examples.
        """
        if batch.text_vec is None:
            return Output()
        self.model.train()
        self.zero_grad()

        # Calculate loss
        labels = self._get_label_tensor(batch)
        scores = self.score(batch)
        loss = self.criterion(scores, labels)
        self.record_local_metric('loss', AverageMetric.many(loss))
        loss = loss.mean()
        self.backward(loss)
        self.update_params()

        # Get predictions
        _, prediction_id = torch.max(scores.float().cpu(), 1)
        preds = [self.class_list[idx] for idx in prediction_id]

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
        _, prediction_id = torch.max(probs.float().cpu(), 1)
        preds = [self.class_list[idx] for idx in prediction_id]

        if batch.labels is None or self.opt['ignore_labels']:
            # interactive mode
            if self.opt.get('print_scores', False):
                preds = self._format_interactive_output(probs, prediction_id)
        else:
            labels = self._get_label_tensor(batch)
            loss = self.criterion(scores, labels)
            self.record_local_metric('loss', AverageMetric.many(loss))

        if self.opt.get('print_scores', False):
            return Output(preds, probs=probs.cpu())
        else:
            return Output(preds)

    def score(self, batch):
        return self.model(*self._model_input(batch))

    def _model_input(self, batch):
        """
        Create the input (x) value for the model.

        If `label_vec` encodes the personalities, which are the target values, it does
        not get passed in as a model input.
        """
        if self.personality_as_label:
            return (batch.text_vec,)
        else:
            return batch.text_vec, batch.label_vec

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Transformer Agent with a contrastive loss.
"""
import torch
from torch.nn import CrossEntropyLoss
from torch.distributions.categorical import Categorical
from typing import Optional, Dict, Union, Tuple, Any
from parlai.core.message import Message

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.agents.fid.fid import FidAgent
from parlai.core.torch_generator_agent import PPLMetric, TorchGeneratorAgent
from parlai.core.metrics import AverageMetric

from parlai.agents.fid.fid import (
    WizIntGoldDocRetrieverFiDAgent,
)
from projects.blenderbot2.agents.blenderbot2 import (
    BlenderBot2FidAgent,
    BlenderBot2FidModel,
    T5BlenderBot2FidModel,
)
from projects.seeker.agents.seeker import (
    ComboFidAgent,
)


class ContrastiveCrossEntropyLoss(CrossEntropyLoss):
    def __init__(
        self,
        ct_loss_weight=1.0,
        num_pos_predictions=1,
        detach_positives_during_ct=False,
        train_ct_on_positive_examples=False,
        train_ce_on_positive_examples=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ct_loss_weight = ct_loss_weight
        self.num_pos_predictions = num_pos_predictions
        self.detach_positives_during_ct = detach_positives_during_ct
        self.train_ct_on_positive_examples = train_ct_on_positive_examples
        self.train_ce_on_positive_examples = train_ce_on_positive_examples

    def __call__(self, x, y, classifier_labels=None, **kwargs):
        if classifier_labels is None:
            classifier_labels = -torch.ones_like(y).to(y.device)

        # turn no-class provided label (-1) into positive label (1)
        classifier_labels_ce = torch.abs(classifier_labels)
        if not self.train_ce_on_positive_examples:
            # only train CE on no-class labels
            classifier_labels_ce = classifier_labels.eq(-1)

        if self.train_ct_on_positive_examples:
            # no-class (-1 to 0), positive (1 to 1), negative (0 to 1)
            classifier_labels_ct = torch.clamp(classifier_labels + 1, max=1)
        else:
            # no-class (-1 to 0), positive (1 to 0), negative (0 to 1)
            classifier_labels_ct = torch.abs(torch.abs(classifier_labels) - 1)
        classifier_labels_ct = classifier_labels_ct.bool()

        ce_loss = super().__call__(x, y, **kwargs)
        # multiply with classifier labels to not train with negative feedback (0)
        ce_loss *= classifier_labels_ce

        # compute the contrastive loss part for the negative labels
        # first, get the positives as the top predictions != target
        preds = torch.topk(x, k=self.num_pos_predictions + 1, axis=-1)
        y_rep = y.unsqueeze(-1).repeat(1, self.num_pos_predictions + 1)
        logits = preds.values - (preds.indices == y_rep) * 1e10

        # if the positive is not in the first k predictions, mask out
        # the final (k+1)'s logit
        prediction_mask = torch.cat(
            (
                torch.zeros_like(logits)[:, :-1],
                torch.abs((preds.indices == y_rep).sum(-1).unsqueeze(1) - 1),
            ),
            1,
        )
        logits -= prediction_mask * 1e10

        # Sample from the categorical distribution of the top-k predictions
        # (with the label masked out).
        preds_dist = Categorical(logits=logits)
        idx_sample = preds_dist.sample()
        sample_preds_values = preds.values[torch.arange(x.shape[0]), idx_sample]

        if self.detach_positives_during_ct:
            sample_preds_values = sample_preds_values.detach()

        # concatenate the logits of the preds with the actual label's logits
        x_target = x[torch.arange(x.shape[0]), y]
        x_ct = torch.cat([x_target.unsqueeze(1), sample_preds_values.unsqueeze(1)], -1)
        # get the y's for the x_ct (the correct label is index 0 if
        # the target is positive and index 1 if the target is negative)
        y_ct = torch.abs(torch.abs(classifier_labels) - 1).type(y.dtype).to(x_ct.device)
        # y_ct = (torch.ones(y.shape) * ).type(y.dtype).to(x_ct.device)
        # compute the contrastive loss as cross entropy loss between x_ct, y_ct
        ct_loss = super().__call__(x_ct, y_ct, **kwargs)
        ct_loss *= classifier_labels_ct

        # remove loss from ignore index
        notnull = y.ne(self.ignore_index)
        ce_loss *= notnull
        ct_loss *= notnull

        loss = ce_loss + self.ct_loss_weight * ct_loss

        return loss, ce_loss, ct_loss, classifier_labels_ce, classifier_labels_ct


class ContrastiveTorchGeneratorAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command line arguments.
        """
        agent = parser.add_argument_group('ContrastiveTorchGeneratorAgent arguments')
        parser.add_argument(
            '--ct-loss-weight',
            type=float,
            help='Coefficient for the contrastive loss (negative examples).',
            default=1.0,
        )
        parser.add_argument(
            '--ct-num-pos-predictions',
            type=int,
            help='How many top predictions do we consider as positives for the contrastive loss?',
            default=1,
        )
        parser.add_argument(
            '--ct-detach-positives',
            type=bool,
            help='If true, we block the gradient for the positives during the contrastive loss.',
            default=False,
        )
        parser.add_argument(
            '--train-ct-on-positive-examples',
            type=bool,
            help='If true, we train with the positive examples in the contrastive loss'
            ' (with the negatives being the top-k sampled from the model).',
            default=False,
        )
        parser.add_argument(
            '--train-ce-on-positive-examples',
            type=bool,
            help='If true, we train with the positive examples in the cross entropy loss.',
            default=True,
        )
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return agent

    def build_criterion(self):
        return ContrastiveCrossEntropyLoss(
            ct_loss_weight=self.opt['ct_loss_weight'],
            num_pos_predictions=self.opt['ct_num_pos_predictions'],
            detach_positives_during_ct=self.opt['ct_detach_positives'],
            ignore_index=self.NULL_IDX,
            train_ct_on_positive_examples=self.opt['train_ct_on_positive_examples'],
            train_ce_on_positive_examples=self.opt.get(
                'train_ce_on_positive_examples', True
            ),
            reduction='none',
        )

    def _v2t(self, vec):
        """
        This method wraps the vec2txt call in a try catch to ensure that sequences with
        generation errors are ignored.

        We return a empty string instead in that scenario.
        """
        try:
            return super()._v2t(vec)
        except AssertionError:
            return ''

    def observe(self, observation: Union[Dict, Message]) -> Message:
        observation = super().observe(observation)
        if 'is_ltr' not in observation:
            observation['is_ltr'] = False
            observation['classifier_label'] = 'none'
            observation['classifier_label_idx'] = -1
            return observation

        classifier_label = observation['classifier_label']
        if classifier_label == 'pos':
            observation['classifier_label_idx'] = 1
        elif classifier_label == 'neg':
            observation['classifier_label_idx'] = 0
        return observation

    def batchify(self, obs_batch, sort=False):
        """
        This method calls the parent class's batchify method and then add
        classifier_label and is_ltr property to the the batch.
        """
        batch = super().batchify(obs_batch, sort=sort)

        if batch.valid_indices is None:
            return batch

        batch.classifier_label = torch.tensor(
            [
                [obs_batch[i].get('classifier_label_idx', -1)]
                for i in batch.valid_indices
            ]
        )
        batch.is_ltr = torch.tensor(
            [[obs_batch[i].get('is_ltr', False)] for i in batch.valid_indices]
        )
        return batch

    def _model_output(self, batch) -> Tuple[Any]:
        return self.model(*self._model_input(batch), ys=batch.label_vec)

    def compute_loss(self, batch, return_output=False):
        if batch.label_vec is None:
            raise ValueError('Cannot compute loss without a label.')
        model_output = self._model_output(batch)
        scores, preds, *_ = model_output
        score_view = scores.reshape(-1, scores.size(-1))
        (loss, ce_loss, ct_loss, ce_mask, ct_mask) = self.criterion(
            score_view,
            batch.label_vec.view(-1),
            batch.classifier_label.repeat(1, scores.shape[1])
            .view(-1)
            .to(batch.label_vec.device),
        )

        def loss_reshape(loss):
            return loss.view(scores.shape[:-1]).sum(dim=1)

        loss = loss_reshape(loss)
        ce_loss = loss_reshape(ce_loss)
        ct_loss = loss_reshape(ct_loss)
        notnull = batch.label_vec.ne(self.NULL_IDX)
        ce_mask = torch.logical_and(notnull, ce_mask.view(-1, batch.label_vec.size(-1)))
        ct_mask = torch.logical_and(notnull, ct_mask.view(-1, batch.label_vec.size(-1)))
        # number of tokens in each examples for cross entropy or cringe loss.
        metric_notnull = torch.logical_or(ce_mask, ct_mask)
        target_tokens = metric_notnull.long().sum(dim=-1)
        ce_target_tokens = ce_mask.long().sum(dim=-1)
        ct_target_tokens = ct_mask.long().sum(dim=-1)

        correct = ((batch.label_vec == preds) * metric_notnull).sum(dim=-1)

        pos_labels = (torch.abs(batch.classifier_label) == 1).view(-1)
        neg_labels = (torch.abs(batch.classifier_label) == 0).view(-1)
        correct_pos = torch.where(pos_labels, correct, -1)
        correct_neg = torch.where(neg_labels, correct, -1)

        # record loss
        self.record_local_metric('loss', AverageMetric.many(loss, target_tokens))
        self.record_local_metric(
            'ce_loss',
            [
                metric if ce_token_cnt > 0 else None
                for ce_token_cnt, metric in zip(
                    ce_target_tokens, AverageMetric.many(ce_loss, target_tokens)
                )
            ],  # type: ignore
        )
        self.record_local_metric(
            'ct_loss',
            [
                metric if ct_token_cnt > 0 else None
                for ct_token_cnt, metric in zip(
                    ct_target_tokens, AverageMetric.many(ct_loss, target_tokens)
                )
            ],  # type: ignore
        )
        # token-wise accuracy
        self.record_local_metric(
            'token_acc',
            [
                metric if per_target_token > 0 else None
                for per_target_token, metric in zip(
                    target_tokens, AverageMetric.many(correct, target_tokens)
                )
            ],  # type: ignore
        )
        self.record_local_metric(
            'token_acc_pos',
            [
                metric if metric >= 0 else None
                for metric in AverageMetric.many(correct_pos, target_tokens)
            ],
        )
        self.record_local_metric(
            'token_acc_neg',
            [
                metric if metric >= 0 else None
                for metric in AverageMetric.many(correct_neg, target_tokens)
            ],
        )
        # perplexity
        self.record_local_metric(
            'ppl_debug',
            [
                metric if per_target_token > 0 else None
                for per_target_token, metric in zip(
                    target_tokens, PPLMetric.many(ce_loss + ct_loss, target_tokens)
                )
            ],  # type: ignore
        )
        self.record_local_metric(
            'ppl_ce',
            [
                metric if ce_token_cnt > 0 else None
                for ce_token_cnt, metric in zip(
                    ce_target_tokens, PPLMetric.many(ce_loss, ce_target_tokens)
                )
            ],  # type: ignore
        )
        self.record_local_metric(
            'ppl_pos',
            [
                metric if pos_label else None
                for pos_label, metric in zip(
                    pos_labels, PPLMetric.many(ce_loss, target_tokens)
                )
            ],
        )
        self.record_local_metric(
            'ppl_ct',
            [
                metric if neg_label else None
                for neg_label, metric in zip(
                    neg_labels, PPLMetric.many(ct_loss, target_tokens)
                )
            ],
        )

        # record sample size
        self.record_local_metric(
            'ce_target_tokens', AverageMetric.many(ce_target_tokens)
        )
        self.record_local_metric(
            'ct_target_tokens', AverageMetric.many(ct_target_tokens)
        )
        self.record_local_metric(
            'total_target_tokens', AverageMetric.many(target_tokens)
        )

        # actually do backwards loss
        loss = loss.sum()
        loss /= target_tokens.sum()  # average loss per token
        if return_output:
            return (loss, model_output)
        else:
            return loss


class ContrastiveTransformerGeneratorAgent(
    ContrastiveTorchGeneratorAgent, TransformerGeneratorAgent
):
    pass


class ContrastiveFidAgent(ContrastiveTorchGeneratorAgent, FidAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command line arguments.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        FidAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser

    def _model_output(self, batch):
        scores, preds, enc_state, *_ = self.get_model_output(batch)
        if scores.size(1) != batch.label_vec.size(1):
            assert self.generation_model == 'bart'
            # ignore start
            scores = scores[:, 1:, :]
            preds = preds[:, 1:]  # type: ignore
        return scores, preds, enc_state


class ContrastiveBB2Agent(ContrastiveFidAgent, BlenderBot2FidAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command line arguments.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        BlenderBot2FidAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser

    def build_model(self) -> BlenderBot2FidModel:
        if self.generation_model == 't5':
            model = T5BlenderBot2FidModel(self.opt, self.dict)
        else:
            model = BlenderBot2FidModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        return model


class ContrastiveBB2WizIntGoldDocRetrieverFiDAgent(
    WizIntGoldDocRetrieverFiDAgent, ContrastiveBB2Agent
):
    pass


class ContrastiveComboFidAgent(ContrastiveFidAgent, ComboFidAgent):
    pass


class ContrastiveComboFidGoldDocumentAgent(
    ContrastiveComboFidAgent, WizIntGoldDocRetrieverFiDAgent
):
    pass

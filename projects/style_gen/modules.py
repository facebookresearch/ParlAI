#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Models and helper classes for style-controlled generation.
"""

import random
from typing import List, Optional

import numpy as np
import torch
from torch import nn as nn

from parlai.agents.transformer.modules import (
    TransformerDecoder,
    TransformerGeneratorModel,
    _normalize,
)
from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.metrics import GlobalAverageMetric, GlobalTimerMetric
from parlai.core.opt import Opt
from parlai.core.torch_classifier_agent import ConfusionMatrixMetric, WeightedF1Metric
from parlai.utils.distributed import is_primary_worker
from parlai.utils.misc import AttrDict, warn_once


STYLE_SEP_TOKEN = ' STYLE '


class StyleAgentMixin:
    """
    Methods for agents that return style from their histories.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.

        Does not add arguments from its superclass because it's a mixin.
        """
        agent = argparser.add_argument_group('StyleAgentMixin arguments')
        agent.add_argument(
            '--use-style-frac',
            type=float,
            default=1.0,
            help='What fraction of the time to use the style label',
        )
        return agent

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.use_style_frac = opt['use_style_frac']

    def get_temp_history(self, observation: Message) -> Optional[str]:
        """
        Conditionally return a style-token string to temporarily insert into history.
        """
        use_style_rand = random.random()
        if use_style_rand < self.use_style_frac:
            # Use the style
            style = observation.get('personality')
            # This key name is dependent on Image-Chat and will change for other tasks.
            # If obs does not contain 'personality' (i.e. at the end of an epoch during
            # validation), there will be no style
        else:
            style = ''
        if style is not None and style != '':
            return STYLE_SEP_TOKEN + style


class ClassifierOnGeneratorModel(TransformerGeneratorModel):
    """
    TransformerGeneratorModel with a classifier head on top of the decoder.

    Useful for performing classification with a pretrained generator model.
    """

    @classmethod
    def build_decoder(
        cls,
        opt,
        dictionary,
        embedding=None,
        padding_idx=None,
        n_positions=1024,
        n_segments=0,
    ):
        """
        Return TransformerDecoderWithEmbeds instead of TransformerDecoder.
        """
        n_layers = (
            opt['n_decoder_layers']
            if opt.get('n_decoder_layers', -1) > 0
            else opt['n_layers']
        )
        return TransformerDecoderWithEmbeds(
            n_heads=opt['n_heads'],
            n_layers=n_layers,
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dictionary),
            embedding=embedding,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=padding_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            n_positions=n_positions,
            activation=opt['activation'],
            variant=opt['variant'],
            n_segments=n_segments,
        )

    def __init__(self, opt, dictionary, num_classes: int, personality_as_label: bool):
        super().__init__(opt, dictionary)
        self.classifier_head = nn.Linear(opt['embedding_size'], num_classes)
        self.personality_as_label = personality_as_label

    def forward(self, *xs):
        """
        Get output class logits from the model.

        :param xs:
            - list of inputs to the encoder/decoder. Elements:
              - text_vec: (LongTensor[bsz, text seqlen])
              - label_vec: (LongTensor[bsz, label seqlen])
                  (Only used if not self.personality_as_label)
        :return:
            - the model's predicted per-class scores.
              (FloatTensor[bsz, len(class_list)])
        """

        if self.personality_as_label:
            # All tokens go into the encoder and classification is learned from that.
            # This is useful in the standard case where we have a fixed utterance that
            # doesn't need to be generated, and we can just stick it all in the encoder
            # to be classified.
            assert len(xs) == 1
            # Only one input allowed
            bsz = xs[0].size(0)
            encoder_states = self.encoder(*xs)
            inputs = self.START.detach().expand(bsz, 1)
            # Generate most likely class given start token as input
            latent, _ = self.decoder(inputs, encoder_states)
            # latent: [bsz, seqlen, emb_dim]
            scores = self.classifier_head(latent.squeeze(dim=1))
        else:
            # Tokens are split between the encoder and decoder and classification is
            # learned from both. This is useful when we want to classify a partially
            # generated utterance, along with its context in the encoder.
            text_vec, label_vec = xs
            encoder_states = self.encoder(text_vec)
            latent, _ = self.decoder(label_vec, encoder_states)
            # latent: [bsz, seqlen, emb_dim]
            scores = self.classifier_head(latent.mean(dim=1))

        return scores


class BatchWithPersonalities(AttrDict):
    """
    Adds a 'personalities' field to the batch in the case where personality information
    is not encoded in any other field.
    """

    def __init__(self, personalities=None, **kwargs):
        super().__init__(personalities=personalities, **kwargs)


class TransformerDecoderWithEmbeds(TransformerDecoder):
    def forward(self, input, encoder_state, embedded_input=None, incr_state=None):
        """
        Forward pass with the ability to pass in token-embedded inputs.
        """

        encoder_output, encoder_mask = encoder_state

        if input is not None:
            seq_len = input.size(1)
            positions = input.new(seq_len).long()
        else:
            seq_len = embedded_input.size(1)
            positions = embedded_input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)

        if incr_state is not None:
            # We're doing incremental decoding, so select only the most recent position
            if input is not None:
                input = input[:, -1:]
            if embedded_input is not None:
                embedded_input = embedded_input[:, -1:, :]
            if positions is not None:
                positions = positions[:, -1:]
        else:
            incr_state = {}

        if embedded_input is not None:
            tensor = embedded_input  # No need to copy because we only reassign below
        else:
            tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        if self.variant == 'xlm':
            tensor = _normalize(tensor, self.norm_embeddings)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, encoder_output, encoder_mask, incr_state
            )
        else:
            for idx, layer in enumerate(self.layers):
                tensor, new_incr_state[idx] = layer(
                    x=tensor,
                    encoder_output=encoder_output,
                    encoder_mask=encoder_mask,
                    incr_state=incr_state.get(idx),
                )

        if self.variant == 'prelayernorm':
            tensor = _normalize(tensor, self.norm_embeddings)

        return tensor, new_incr_state


class ClassificationMixin(Agent):
    """
    Mixin for adding classification metrics to non-classifier models.
    """

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        if shared is None:
            self.reset_metrics()

    def _update_confusion_matrix(
        self, predictions: List[Optional[str]], labels: List[Optional[List[str]]]
    ):
        """
        Update the confusion matrix given the batch and predictions.

        :param predictions: List of strings of length batchsize. Each string is a label
            predicted by the classifier. A string will be None if the corresponding
            observation is empty.
        :param labels: List of label fields from the observations. Fields may be Nones
            if the observations are empty.
        """
        f1_dict = {}
        explode_labels = []
        for x in labels:
            if x is not None and len(x) > 0:
                assert len(x) == 1, 'Multiple labels are not currently supported!'
                explode_labels.append(x[0])
            else:
                explode_labels.append(None)

        # Check that predictions and labels have Nones in the same places, and then
        # filter the Nones out because we can't compute metrics with them
        assert len(predictions) == len(labels)
        assert all(
            [
                (pred is None and label is None)
                or (pred is not None and label is not None)
                for pred, label in zip(predictions, explode_labels)
            ]
        )
        filtered_predictions = [pred for pred in predictions if pred is not None]
        filtered_labels = [label for label in explode_labels if label is not None]

        class_list = set(filtered_predictions + filtered_labels)
        for class_name in class_list:
            prec_str = f'class_{class_name}_prec'
            recall_str = f'class_{class_name}_recall'
            f1_str = f'class_{class_name}_f1'
            precision, recall, f1 = ConfusionMatrixMetric.compute_metrics(
                filtered_predictions, filtered_labels, class_name
            )
            f1_dict[class_name] = f1
            self.record_local_metric(prec_str, precision)
            self.record_local_metric(recall_str, recall)
            self.record_local_metric(f1_str, f1)
        self.record_local_metric('weighted_f1', WeightedF1Metric.compute_many(f1_dict))

    def _get_preds(self, batch_reply):
        preds = [reply.get('text') for reply in batch_reply]
        if all(x is None for x in preds):
            return None

        return preds

    def _get_labels(self, observations, labels_field: str):
        labels = [obs.get(labels_field) for obs in observations]
        return labels

    def batch_act(self, observations):

        # clear local metrics before anything else
        self._local_metrics.clear()

        # initialize a list of replies with this agent's id
        batch_reply = [
            Message({'id': self.getID(), 'episode_done': False}) for _ in observations
        ]

        # check if there are any labels available, if so we will train on them
        self.is_training = any('labels' in obs for obs in observations)

        # create a batch from the vectors
        batch = self.batchify(observations)
        self.global_metrics.add('exps', GlobalTimerMetric(batch.batchsize))

        if (
            'label_vec' in batch
            and 'text_vec' in batch
            and batch.label_vec is not None
            and batch.text_vec is not None
        ):
            # tokens per batch
            # we divide by the binary is_primary_worker() so that the numerator is
            # num_tokens in all workers, and the denominator is 1.
            lt = (batch.label_vec != self.NULL_IDX).sum().item()
            ltpb = GlobalAverageMetric(lt, float(is_primary_worker()))
            self.global_metrics.add('ltpb', ltpb)
            self.global_metrics.add('ltps', GlobalTimerMetric(lt))

            ct = (batch.text_vec != self.NULL_IDX).sum().item()
            ctpb = GlobalAverageMetric(ct, float(is_primary_worker()))
            self.global_metrics.add('ctpb', ctpb)
            self.global_metrics.add('ctps', GlobalTimerMetric(ct))

            ttpb = GlobalAverageMetric(ct + lt, float(is_primary_worker()))
            self.global_metrics.add('tpb', ttpb)
            self.global_metrics.add('tps', GlobalTimerMetric(ct + lt))

        if self.is_training:
            # register the start of updates for later counting when they occur
            self.global_metrics.add('ups', GlobalTimerMetric(0))
            output = self.train_step(batch)
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back gradients.
                output = self.eval_step(batch)

        if output is not None:
            # local metrics are automatically matched up
            self.match_batch(batch_reply, batch.valid_indices, output)

        # broadcast the metrics back
        for k, values in self._local_metrics.items():
            if len(values) != len(batch.valid_indices):
                raise IndexError(
                    f"Batchsize mismatch on metric {k} (got {len(values)}, "
                    f"expected {len(batch.valid_indices)}"
                )
            for i, value in zip(batch.valid_indices, values):
                if 'metrics' not in batch_reply[i]:
                    batch_reply[i]['metrics'] = {}
                batch_reply[i]['metrics'][k] = value

        # register the end of timers
        endtimer = GlobalTimerMetric(0)
        self.global_metrics.add('exps', endtimer)
        if (
            'label_vec' in batch
            and 'text_vec' in batch
            and batch.label_vec is not None
            and batch.text_vec is not None
        ):
            self.global_metrics.add('ltps', GlobalTimerMetric(0))
            self.global_metrics.add('ctps', GlobalTimerMetric(0))
            self.global_metrics.add('tps', GlobalTimerMetric(0))

        preds = self._get_preds(batch_reply)
        if 'labels' in observations[0]:
            labels_field = 'labels'
        elif 'eval_labels' in observations[0]:
            labels_field = 'eval_labels'
        else:
            labels_field = None

        if preds is not None and labels_field is not None:
            labels_lst = self._get_labels(observations, labels_field)
            self._update_confusion_matrix(preds, labels_lst)

        return batch_reply

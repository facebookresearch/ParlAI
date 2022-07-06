#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Models and helper classes for style-controlled generation.
"""

from parlai.core.params import ParlaiParser
import random
from typing import List, Optional

import numpy as np
import torch
from torch import nn as nn

from parlai.agents.transformer.modules import (
    TransformerDecoder,
    TransformerGeneratorModel,
)
from parlai.core.agents import Agent
from parlai.core.message import Message
from parlai.core.opt import Opt
from parlai.core.torch_classifier_agent import ConfusionMatrixMetric, WeightedF1Metric
from parlai.utils.misc import warn_once


STYLE_SEP_TOKEN = ' STYLE '


class StyleAgentMixin:
    """
    Methods for agents that return style from their histories.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.

        Does not add arguments from its superclass because it's a mixin.
        """
        agent = parser.add_argument_group('StyleAgentMixin arguments')
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
    def build_decoder(cls, opt, embedding=None, **kwargs):
        """
        Return TransformerDecoderWithEmbeds instead of TransformerDecoder.
        """
        return TransformerDecoderWithEmbeds(opt=opt, embedding=embedding, **kwargs)

    def __init__(self, opt, dictionary, num_classes: int):
        super().__init__(opt, dictionary)
        self.classifier_head = nn.Linear(opt['embedding_size'], num_classes)

    def forward(self, *xs):
        """
        Get output class logits from the model.

        :param xs:
            - list of inputs to the encoder/decoder. Elements:
              - text_vec: (LongTensor[bsz, text seqlen])
        :return:
            - the model's predicted per-class scores.
              (FloatTensor[bsz, len(class_list)])
        """

        # All tokens go into the encoder and classification is learned from that.
        assert len(xs) == 1
        # Only one input allowed
        bsz = xs[0].size(0)
        encoder_states = self.encoder(*xs)
        inputs = self.START.detach().expand(bsz, 1)
        # Generate most likely class given start token as input
        latent, _ = self.decoder(inputs, encoder_states)
        # latent: [bsz, seqlen, emb_dim]
        scores = self.classifier_head(latent.squeeze(dim=1))

        return scores


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
            tensor = self.norm_embeddings(tensor)
        if positions.max().item() > self.n_positions:
            warn_once(
                'You are inputting a sequence of {x} length, but only have '
                '--n-positions {y}. Set --truncate or increase --n-positions'.format(
                    x=positions.max().item(), y=self.n_positions
                )
            )
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        if self.variant == 'bart':
            tensor = self.norm_embeddings(tensor)

        tensor = self.dropout(tensor)  # --dropout

        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, encoder_output, encoder_mask, incr_state=incr_state
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
            tensor = self.norm_embeddings(tensor)

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

        # Check that predictions and labels have Nones in the same places, and then
        # filter the Nones out because we can't compute metrics with them
        assert len(predictions) == len(labels)
        assert all(
            [
                (pred is None and label is None)
                or (pred is not None and label is not None)
                for pred, label in zip(predictions, labels)
            ]
        )
        filtered_predictions = [pred for pred in predictions if pred is not None]
        filtered_labels = [label for label in labels if label is not None]

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

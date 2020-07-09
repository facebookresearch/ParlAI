#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Models and helper classes for style-controlled generation.
"""

import random

import numpy as np
import torch
from torch import nn as nn

from parlai.agents.transformer.modules import (
    TransformerDecoder,
    TransformerGeneratorModel,
    _normalize,
)
from parlai.core.torch_agent import History
from parlai.utils.misc import AttrDict, warn_once


STYLE_SEP_TOKEN = ' STYLE '


class StyleHistoryMixin(History):
    """
    Methods for adding style to history.
    """

    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        self.use_style_frac = opt['use_style_frac']
        self.style = None

    def reset(self):
        super().reset()
        self.style = None

    def update_history(self, obs, *args, **kwargs):
        super().update_history(obs, *args, **kwargs)
        use_style_rand = random.random()
        if use_style_rand < self.use_style_frac:
            # Use the style
            self.style = obs.get('personality')
            # This key name is dependent on ImageChat and will change for other tasks.
            # If obs does not contain 'personality' (i.e. at the end of an epoch during
            # validation), there will be no style
            if self.style == '':
                self.style = None
        else:
            self.style = None

    def get_history_str(self):
        history_str = super().get_history_str()
        if history_str is not None and self.style is not None:
            history_str += STYLE_SEP_TOKEN + self.style

        return history_str

    def get_history_vec(self):
        history = super().get_history_vec()

        if history is not None and self.style is not None:
            style = STYLE_SEP_TOKEN + self.style
            style_tok = self.parse(style)
            if self.vec_type == 'deque':
                history.extend(style_tok)
            else:
                history += style_tok

        return history


class StyleAgentMixin:
    """
    Methods for agents that return style from their histories.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent. Does not add arguments
        from its superclass because it's a mixin.
        """
        agent = argparser.add_argument_group('Style arguments')
        agent.add_argument(
            '--use-style-frac',
            type=float,
            default=1.0,
            help='What fraction of the time to use the style label',
        )
        return agent


class StyleHistory(StyleHistoryMixin, History):
    """
    Modify history to save the style.
    """


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
            # learned from both.
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
        # TODO: perhaps reduce the amount of code duplicated from TransformerDecoder.
        #  This would require modularizing several snippets of code inside
        #  TransformerDecoder methods.

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

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import torch
from typing import Dict, Optional, Tuple
import torch.nn as nn

from parlai.agents.transformer.modules import (
    TransformerDecoder,
    TransformerGeneratorModel,
)

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
import torch.nn.functional as F

###########################################
#         Hash Ladder Transformer         #
###########################################

"""
Use with, e.g.:

parlai train_model -m projects.params_vs_compute.hash_ladder.hash_ladder:HashLadderAgent -t convai2:normalized -mf /tmp/model_file --ladder-size 1 --hash-size 32 --hash-layer 1
"""


class HashLadderAgent(TransformerGeneratorAgent):
    """
    Simple implementation of Hash Layers and the Ladder model from the following papers:

    - https://arxiv.org/abs/2106.04426
    - https://arxiv.org/abs/2106.04279
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        # Add transformer args.
        parser.add_argument(
            '--ladder-size',
            type=int,
            default=1,
            help='Number of ladder steps, default is not to use ladder.',
        )
        parser.add_argument(
            '--hash-size', type=int, default=32, help='Number of hash bins.'
        )
        parser.add_argument(
            '--hash-layer',
            type=int,
            default=7,
            help='Layer number the Hash Layer appears on.',
        )
        return parser

    def build_model(self, states=None):
        wrapped_class = TransformerGeneratorModel.with_components(decoder=Decoder)
        return wrapped_class(self.opt, self.dict)

    def dummy_loss(self):
        """
        Hack from Guillaume to fix adaptive weights with distributed code.
        """
        if hasattr(self.model, 'module'):
            ffn = self.model.module.decoder.layers[self.opt['hash_layer']].ffn
        else:
            ffn = self.model.decoder.layers[self.opt['hash_layer']].ffn
        dummy_loss = 0 * (
            sum(x.weight[0, 0] for x in ffn.linears1)
            + sum(x.weight[0, 0] for x in ffn.linears2)
            + sum(x.bias[0] for x in ffn.linears1)
            + sum(x.bias[0] for x in ffn.linears2)
        )
        return dummy_loss

    def compute_loss(self, batch, return_output=False):
        """
        Compute and return the loss for the given batch.
        """
        if return_output:
            loss, model_output = super().compute_loss(batch, return_output)
        else:
            loss = super().compute_loss(batch, return_output)

        if self.opt['hash_layer'] != -1:
            loss = loss + self.dummy_loss()

        if return_output:
            return (loss, model_output)
        else:
            return loss


class Decoder(TransformerDecoder):
    """
    Custom Decoder with Ladder model.
    """

    def build_layers(self) -> nn.ModuleList:
        # HACK: Adding vocab size to opt for use in HashLayerFFN
        self.opt['dict_size'] = self.embeddings.weight.size(0)
        layers = nn.ModuleList()
        for i in range(self.n_layers):
            layer_class = self.swappables.layer
            if self.opt['hash_layer'] == i:
                layer_class = layer_class.with_components(feedforward=HashLayerFFN)
            layers.append(
                layer_class(
                    self.opt,
                    attention_dropout=self.opt.get('attention_dropout', 0.0),
                    relu_dropout=self.opt.get('relu_dropout', 0.0),
                    dropout=self.opt.get('dropout', 0.0),
                    activation=self.activation,
                    variant=self.variant,
                )  # type: ignore
            )
        return layers

    def forward_layers(
        self,
        tensor: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        incr_state: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Override of forward_layers of TransformerDecoder.
        """
        new_incr_state = {}
        for _s in range(0, self.opt['ladder_size']):
            tensor, new_incr_state = super().forward_layers(
                tensor=tensor,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
                incr_state=incr_state,
                **kwargs,
            )
        return tensor, new_incr_state

    def forward(
        self,
        input: torch.Tensor,
        encoder_state,
        incr_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Overrides TransformerDecoder forward.
        """
        return super().forward(
            input, encoder_state, incr_state=incr_state, orig_input=input, **kwargs
        )


class HashLayerFFN(nn.Module):
    """
    Implements the Hash Layer FFN.
    """

    def __init__(self, opt, dim, dim_hidden, relu_dropout=0, activation='relu'):
        super(HashLayerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.nonlinear = F.relu
        self.opt = opt
        self.dim = dim
        self.dim_hidden = dim_hidden
        self.hashsize = opt['hash_size']

        linears1 = []
        linears2 = []

        for i in range(0, self.hashsize):
            linears1.append(nn.Linear(dim, dim_hidden))
            nn.init.xavier_uniform_(linears1[i].weight)
        for i in range(0, self.hashsize):
            linears2.append(nn.Linear(dim_hidden, dim))
            nn.init.xavier_uniform_(linears2[i].weight)

        self.linears1 = nn.ModuleList(linears1)
        self.linears2 = nn.ModuleList(linears2)

    def hash(self, xi):
        # Insert your choice of hash function here.
        # In this code we simply randomly hash based on the given token IDs for simplicity.
        if not hasattr(self, 'hash_bin_map'):
            # create random mapping.
            sz = self.opt['dict_size']
            self.hash_bin_map = torch.LongTensor(sz).fill_(0)
            import random

            random.seed(42)
            for i in range(sz):
                self.hash_bin_map[i] = random.randrange(0, self.hashsize)

        # Now compute the hash bins given the mapping function (Whatever it is).
        return self.hash_bin_map[xi]

    def forward(self, x, orig_input):
        """
        Forward pass.
        """
        xhs = self.hash(orig_input)

        # Now do the real work.
        # This implementation could be more efficient -- but it works.
        index_list = [
            torch.eq(xhs, i).nonzero(as_tuple=True) for i in range(self.hashsize)
        ]
        final_output = x.new_zeros(x.shape)

        for i in range(self.hashsize):
            vecs = x[index_list[i][0], index_list[i][1], :]
            if vecs.shape[0] > 0:
                x1 = self.linears1[i](vecs)
                x1 = self.nonlinear(x1)
                x1 = self.relu_dropout(x1)  # --relu-dropout
                x1 = self.linears2[i](x1)
                final_output[index_list[i][0], index_list[i][1], :] = x1

        return final_output

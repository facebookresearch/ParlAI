#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations
import torch
from typing import Optional, Tuple
import torch.nn as nn

from parlai.agents.transformer.modules import (
    TransformerDecoder,
    TransformerGeneratorModel,
)
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser


###########################################
# Transformer With Encoder Swapped #
###########################################

"""
Use with, e.g.:

parlai train_model -m projects.params_vs_compute.hash_ladder.hash_ladder:HashLadderAgent -t convai2:normalized -mf /tmp/model_file
"""


class HashLadderAgent(TransformerGeneratorAgent):
    """
    Simple implementation of Hash Layers and the Ladder model from the following papers:
    
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        TransformerGeneratorAgent.add_cmdline_args(
            parser, partial_opt=partial_opt
        )  # add transformer args
        parser.add_argument(
            '--ladder-size',
            type=int,
            default=1,
            help='Number of ladder steps, default is not to use ladder.',
        )
        return parser

    def build_model(self, states=None):
        wrapped_class = TransformerGeneratorModel.with_components(
            # encoder=ShiftInvariantEncoder,
            decoder=Decoder
        )
        return wrapped_class(self.opt, self.dict)


class Decoder(TransformerDecoder):
    """
    Custom Decoder with Ladder model
    """

    def forward_layers(
        self,
        tensor: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
        incr_state: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of decoder layers.

        :param tensor:
            embedded input tensor for the decoder
        :param enc_out:
            encoder outputs
        :param enc_mask:
            encoder output mask
        :param incr_state:
            Dict mapping layer_idx to incremental state

        :return (tensor, new_incr_state):
            return encoding after applying decoder layers, as well
            as new incremental decoding state.
        """
        new_incr_state = {}
        if getattr(self.layers, 'is_model_parallel', False):
            tensor, new_incr_state = self._apply_model_parallel(
                tensor, encoder_output, encoder_mask, incr_state
            )
        else:
            for s in range(0, self.opt['ladder_size']):
                for idx, layer in enumerate(self.layers):
                    tensor, new_incr_state[idx] = layer(
                        x=tensor,
                        encoder_output=encoder_output,
                        encoder_mask=encoder_mask,
                        incr_state=incr_state.get(idx),
                    )

        return tensor, new_incr_state

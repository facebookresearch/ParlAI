#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from parlai.agents.transformer.transformer import add_common_cmdline_args
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.torch import padded_tensor

from .modules import (
    PassThroughEncoder,
    TransformerDecoderOnly,
    TransformerGeneratorModel,
)


class DecoderOnlyAgent(TorchGeneratorAgent):
    """
    DecoderOnlyAgent.

    Implementation of TorchGeneratorAgent, where the model is a Decoder-Only Transformer.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        agent = parser.add_argument_group('Decoder-Only Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(parser, partial_opt=partial_opt)

        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return agent

    def build_model(self, states=None):
        """
        Override of ``TorchAgent.build_model``.
        """
        assert (
            self.opt['n_encoder_layers'] == -1
        ), "Decoder-only model cannot have encoder layers."
        wrapped_class = TransformerGeneratorModel.with_components(
            encoder=PassThroughEncoder, decoder=TransformerDecoderOnly
        )
        return wrapped_class(self.opt, self.dict)

    def _pad_tensor(self, items, is_label=False):
        """
        Override of ``TorchAgent._pad_tensor``.
        """
        if is_label:
            return padded_tensor(
                items, pad_idx=self.NULL_IDX, left_padded=False, fp16friendly=False
            )
        else:
            return padded_tensor(
                items, pad_idx=self.NULL_IDX, left_padded=True, fp16friendly=False
            )

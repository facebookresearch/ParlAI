#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

from parlai.agents.transformer.transformer import add_common_cmdline_args
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.torch import IdentityLayer, padded_tensor

from .modules import TransformerDecoderOnly, TransformerGeneratorModel


class DecoderOnlyAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        agent = parser.add_argument_group("Transformer Decoder Only Args")
        add_common_cmdline_args(agent)
        agent.add_argument(
            "--add-special-tokens",
            type=bool,
            default=True,
            help="Add special tokens (like PAD, etc.). If False, "
            "Can only use with batch size 1.",
        )
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        return parser

    # maybe for quick-and-dirty I could just use a SentencePiece dictionary here
    # @staticmethod
    # def dictionary_class():
    #     return Gpt2DictionaryAgent

    def build_model(self, states=None):
        """
        Build and return model.
        """
        assert (
            self.opt['n_encoder_layers'] == -1
        ), "Decoder-only model cannot have encoder layers."
        wrapped_class = TransformerGeneratorModel.with_components(
            encoder=IdentityLayer, decoder=TransformerDecoderOnly
        )
        return wrapped_class(self.opt, self.dict)

    def _pad_tensor(self, items, is_label=False):
        if is_label:
            return padded_tensor(
                items, pad_idx=self.NULL_IDX, left_padded=False, fp16friendly=False
            )
        else:
            return padded_tensor(
                items, pad_idx=self.NULL_IDX, left_padded=True, fp16friendly=False
            )

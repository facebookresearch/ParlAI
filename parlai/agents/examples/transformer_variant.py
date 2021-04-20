#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Example code for specifying custom transformer variants.

TransformerVariantAgent:
- Minimal changes needed to:
    - Swap out a high-level component (encoder)
    - Swap out a low-level component (decoder.layer.self_attention)

VerboseTransformerAgent:
- Doesn't swap out anything
- Fully-specifies all components, for illustration

ConfigurableTransformerAgent:
- Swaps out implementations based on command line args
"""
from __future__ import annotations
import torch
from enum import Enum
from typing import Optional, Tuple, Union

from parlai.agents.transformer.modules import (
    TransformerFFN,
    MultiHeadAttention,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerGeneratorModel,
)
from parlai.agents.transformer.modules.interfaces import ModularComponentBuilder
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
import parlai.utils.logging as logging


###########################################
# Transformer With Two Components Swapped #
###########################################


class TransformerVariantAgent(TransformerGeneratorAgent):
    """
    Swapping out two things:
    1) Encoder (high-level component)
    2) Decoder self attention (low-level component)
    """

    def build_model(self, states=None):
        template = TransformerGeneratorModel.Template(
            encoder=ModularComponentBuilder(MyCustomEncoder),
            decoder=ModularComponentBuilder(
                TransformerDecoder,
                TransformerDecoder.Template(
                    layer=ModularComponentBuilder(
                        TransformerDecoderLayer,
                        TransformerDecoderLayer.Template(
                            self_attention=MyCustomAttention
                        ),
                    )
                ),
            ),
        )
        return TransformerGeneratorModel(self.opt, self.dict, template)


class MyCustomEncoder(TransformerEncoder):
    """
    For brevity this subclasses TransformerEncoder, but you could
    write your own nn.Module from scratch as long as the __init__
    and forward signatures match TransformerEncoder.
    """

    def forward(  # type: ignore
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor]]:
        # Comment out the following line and write your custom `forward` instead.
        return super().forward(input, positions, segments)


class MyCustomAttention(MultiHeadAttention):
    """
    For brevity this just renames MultiHeadAttention, but
    ideally you'd define a new nn.Module with the same
    __init__ and forward signature as MultiHeadAttention
    """

    pass


#######################################
# Fully-specified Default Transformer #
#######################################


class VerboseTransformerAgent(TransformerGeneratorAgent):
    """
    Doesn't make any changes to TransformerGeneratorModel, just
    specifies all subcomponents explicitly as a refernce of what's
    swappable.
    """

    def build_model(self, states=None):
        transformer_template = TransformerGeneratorModel.Template(
            encoder=ModularComponentBuilder(
                TransformerEncoder,
                TransformerEncoder.Template(
                    layer=ModularComponentBuilder(
                        TransformerEncoderLayer,
                        TransformerEncoderLayer.Template(
                            self_attention=MultiHeadAttention,
                            feedforward=TransformerFFN,
                        ),
                    )
                ),
            ),
            decoder=ModularComponentBuilder(
                TransformerDecoder,
                TransformerDecoder.Template(
                    layer=ModularComponentBuilder(
                        TransformerDecoderLayer,
                        TransformerDecoderLayer.Template(
                            encoder_attention=MultiHeadAttention,
                            self_attention=MultiHeadAttention,
                            feedforward=TransformerFFN,
                        ),
                    )
                ),
            ),
        )
        return TransformerGeneratorModel(
            opt=self.opt, dictionary=self.dict, template=transformer_template
        )


################################################
# Command-line Configurable Custom Transformer #
################################################


class DecoderFeedForwardVariant(Enum):
    ONE = 'one'
    TWO = 'two'


class DecoderFFNOne(TransformerFFN):
    logging.info("Using Decoder FFN Variant One")


class DecoderFFNTwo(TransformerFFN):
    logging.info("Using Decoder FFN Variant Two")


class ConfigurableTransformerAgent(TransformerGeneratorAgent):
    """
    Illustrates swapping out components based on command line args.
    Specifically, swaps out the decoder ffn between two options.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        agent = parser.add_argument_group('MyCustom Transformer Arguments')
        parser.add_argument(
            '--decoder-ffn-variants',
            type=DecoderFeedForwardVariant,
            default=DecoderFeedForwardVariant.ONE,
            help='Some variants in the decoder FFN implementation',
        )
        return agent

    def build_model(self, states=None):
        decoder_variant: DecoderFeedForwardVariant = self.opt['decoder_ffn_variants']
        if decoder_variant == DecoderFeedForwardVariant.ONE:
            decoder_ffn_class = DecoderFFNOne
        elif decoder_variant == DecoderFeedForwardVariant.TWO:
            decoder_ffn_class = DecoderFFNTwo
        else:
            logging.error(
                'Invalid --decoder-ffn-variants option, defaulting to original ffn implementation.'
            )
            decoder_ffn_class = TransformerFFN

        transformer_template = TransformerGeneratorModel.Template(
            decoder=ModularComponentBuilder(
                TransformerDecoder,
                TransformerDecoder.Template(
                    layer=ModularComponentBuilder(
                        TransformerDecoderLayer,
                        TransformerDecoderLayer.Template(feedforward=decoder_ffn_class),
                    )
                ),
            )
        )
        return TransformerGeneratorModel(
            opt=self.opt, dictionary=self.dict, template=transformer_template
        )

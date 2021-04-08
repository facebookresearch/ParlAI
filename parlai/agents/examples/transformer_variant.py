#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from parlai.agents.transformer.modules.decoder import TransformerDecoder
from parlai.agents.transformer.modules.interfaces import ModularComponentSpec
import torch
from typing import Optional, Tuple, Union

from parlai.agents.transformer.modules import (
    MultiHeadAttention,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerGeneratorModel,
)
from parlai.agents.transformer.transformer import TransformerGeneratorAgent


class TransformerVariantAgent(TransformerGeneratorAgent):
    def build_model(self, states=None):
        # Using a custom encoder and a custom decoder self attention
        template = TransformerGeneratorModel.Template(
            encoder=ModularComponentSpec(
                klass=MyCustomEncoder, template=TransformerEncoder.Template()
            ),
            decoder=ModularComponentSpec(
                klass=TransformerDecoder,
                template=TransformerDecoder.Template(
                    layer=ModularComponentSpec(
                        klass=TransformerDecoderLayer,
                        template=TransformerDecoderLayer.Template(
                            self_attention=MyCustomAttention
                        ),
                    )
                ),
            ),
        )
        return TransformerGeneratorModel(self.opt, self.dict, template)


class MyCustomEncoder(TransformerEncoder):
    def forward(  # type: ignore
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor]]:
        # Comment out the following line and write your custom `forward` instead.
        return super().forward(input, positions, segments)


class MyCustomAttention(MultiHeadAttention):
    # For brevity this subclasses MultiHeadAttention, but
    # ideally you'd define a new nn.Module with the same
    # __init__ and forward signature as MultiHeadAttention
    pass

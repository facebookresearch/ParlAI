#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import replace
import torch
from typing import Tuple, Optional, Union

from parlai.agents.transformer.modules import (
    TransformerGeneratorModel,
    TransformerEncoder,
)
from parlai.agents.transformer.transformer import TransformerGeneratorAgent


class TransformerVariantAgent(TransformerGeneratorAgent):
    def build_model(self, states=None):
        manifest = TransformerGeneratorModel.Manifest()
        manifest.encoder = replace(manifest.encoder, klass=MyCustomEncoder)
        return TransformerGeneratorModel(self.opt, self.dict, manifest)


class MyCustomEncoder(TransformerEncoder):
    def forward(  # type: ignore
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.BoolTensor]]:
        # Comment out the following line and write your custom `forward` instead.
        return super().forward(input, positions, segments)

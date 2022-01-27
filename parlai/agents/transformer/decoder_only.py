#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.torch import IdentityLayer, padded_tensor

from .modules import TransformerDecoderOnly, TransformerGeneratorModel


class DecoderOnlyAgent(TorchGeneratorAgent):
    def build_model(self, states=None):
        """
        Override of ``TorchAgent.build_model``.
        """
        assert (
            self.opt['n_encoder_layers'] == -1
        ), "Decoder-only model cannot have encoder layers."
        wrapped_class = TransformerGeneratorModel.with_components(
            encoder=IdentityLayer, decoder=TransformerDecoderOnly
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

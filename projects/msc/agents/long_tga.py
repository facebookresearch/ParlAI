#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Optional, Tuple

from parlai.agents.transformer.modules import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerGeneratorModel,
)
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser


###########################################
# Transformer With Encoder Swapped #
###########################################


class TransformerVariantAgent(TransformerGeneratorAgent):
    """
    Swapping out Encoder:
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        TransformerGeneratorAgent.add_cmdline_args(
            parser, partial_opt=partial_opt
        )  # add transformer args
        parser.add_argument(
            '--n-positions-init',
            type=int,
            default=None,
            hidden=True,
            help='Number of positional embeddings from the init model. Defaults '
            'to truncate or 1024 if not provided.',
        )
        return parser

    def build_model(self, states=None):
        wrapped_class = TransformerGeneratorModel.with_components(
            encoder=ShiftInvariantEncoder, decoder=TransformerDecoder
        )
        return wrapped_class(self.opt, self.dict)


class ShiftInvariantForwardEmbeddingMixin:
    """
    Custom Encoder with extended position embedding that is shift-invariant.
    """

    def __init__(self, opt, *args, **kwargs):
        super().__init__(opt, *args, **kwargs)
        self.n_positions_init = opt.get("n_positions_init", None)

    def forward_embedding(
        self,
        input: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
        segments: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """
        Embed tokens prior to feeding into transformer.

        :param LongTensor[batch,seqlen] input:
            The input IDs
        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs
        :param LongTensor[batch,seqlen]:
            If provided, additionally adds ``segments`` as extra embedding features.

        :return (tensor, mask):
            return embedded input and mask
        """
        if self.n_positions_init is not None:
            mask = input != self.padding_idx
            if positions is None:
                positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
            positions = self.rearrange_positions(positions)  # type: ignore
        return super().forward_embedding(input, positions, segments)

    def rearrange_positions(self, positions: torch.LongTensor) -> torch.LongTensor:
        """
        Rearange positions prior to feeding into transformer so that the existing ones,
        i.e. the first self.n_positions_init, do not change from before.

        :param LongTensor[batch,seqlen] positions:
            Positions for input IDs

        :return LongTensor[batch,seqlen] new_positions
            return positions that's shift-invariant
        """
        # max positions per sample in each batch, return batch * 1
        if self.n_positions <= self.n_positions_init:
            return positions
        pos_offset = (
            (torch.max(positions, dim=1).values - self.n_positions_init + 1)
            .unsqueeze(1)
            .clamp(min=0)
        )
        # shift the positions to the right by (max_position - self.original_n_positions).clamp_(min=0)
        # notice that remainder always return non-negative values (unlike torch.fmod), e.g. torch.remainder([-1, -2], 10) = [9, 8]
        return torch.remainder(positions - pos_offset, self.n_positions)  # type: ignore


class ShiftInvariantEncoder(ShiftInvariantForwardEmbeddingMixin, TransformerEncoder):
    pass

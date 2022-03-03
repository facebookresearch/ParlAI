#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from parlai.agents.transformer.transformer import add_common_cmdline_args
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.torch_generator_agent import TorchGeneratorAgent
from parlai.utils.logging import logging
from parlai.utils.misc import recursive_getattr
from parlai.utils.torch import padded_tensor

from .modules import (
    PassThroughEncoder,
    TransformerDecoderOnly,
    TransformerGeneratorModel,
)


class DecoderAgent(TorchGeneratorAgent):
    """
    DecoderOnlyAgent.

    Implementation of TorchGeneratorAgent, where the model is a Decoder-Only
    Transformer.
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

        Pads context tensor on the left and label tensor on the right, such that when
        they are concatenated the example meets in the middle to form a continuous
        sequence.
        """
        return padded_tensor(
            items,
            pad_idx=self.NULL_IDX,
            left_padded=(not is_label),
            fp16friendly=self.fp16,
        )

    def _resize_token_embeddings(self, state_dict, msg=None):
        """
        Resize the token embeddings when adding extra special tokens.
        """
        # map extra special tokens carefully
        new_size = self.model.embeddings.weight.size()[0]
        orig_size = state_dict['embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {orig_size} to {new_size}')
        if new_size <= orig_size:
            # new size should be greater than original size,
            # as we are adding special tokens
            raise RuntimeError(msg)

        for emb_weights in ['embeddings.weight', 'decoder.embeddings.weight']:
            # get new_embs
            old_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.model, emb_weights).to(old_embs.device)
            # copy over old weights
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            # reset in state dict
            state_dict[emb_weights] = new_embs

        return state_dict

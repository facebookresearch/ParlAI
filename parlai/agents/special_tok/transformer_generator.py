#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.agents.transformer.transformer import TransformerGeneratorAgent as Base
from .agents import SpecialTokenMixin, add_common_args, recursive_getattr

from parlai.utils.logging import logging


class TransformerGeneratorAgent(SpecialTokenMixin, Base):
    """
    TransformerGeneratorAgent with special tokens added.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        argparser = add_common_args(argparser)
        argparser = super(cls, TransformerGeneratorAgent).add_cmdline_args(argparser)
        return argparser

    def _resize_token_embeddings(self, state_dict, msg=None):
        # map extra special tokens carefully
        new_size = self.model.embeddings.weight.size()[0]
        orig_size = state_dict['embeddings.weight'].size()[0]
        logging.info(f'Resizing token embeddings from {orig_size} to {new_size}')
        if new_size <= orig_size:
            # new size should be greater than original size,
            # as we are adding special tokens
            raise RuntimeError(msg)

        for emb_weights in [
            'embeddings.weight',
            'encoder.embeddings.weight',
            'decoder.embeddings.weight',
        ]:
            # get new_embs
            old_embs = state_dict[emb_weights]
            new_embs = recursive_getattr(self.model, emb_weights).to(old_embs.device)
            # copy over old weights
            new_embs.data[:orig_size, :] = old_embs.data[:orig_size, :]
            # reset in state dict
            state_dict[emb_weights] = new_embs

        # now try loading again
        self.model.load_state_dict(state_dict)

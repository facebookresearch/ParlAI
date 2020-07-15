#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.utils.logging import logging

import functools
import torch
from typing import Dict, Any, List

SPECIAL_TOKS = 'PARTY,PARROT'


def recursive_getattr(obj, attr, *args):
    """
    Recursive call to getattr for nested attributes
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def add_common_args(argparser):
    """
    Add cmdline args
    """
    argparser.add_argument(
        '--special-tok-lst',
        type=str,
        default=SPECIAL_TOKS,
        help='Comma separated list of special tokens'
    )

    return argparser


class SpecialTokenMixin:
    """
    Mixin adding special tokens to the dictionary.
    """
    def _get_special_tokens(self) -> List[str]:
        """
        Return list of special tokens.

        Made easily overridable for special cases.
        """
        return self.opt['special_tok_lst'].split(',')

    def build_dictionary(self):
        """
        Return the constructed dictionary, which will be set to self.dict.

        If you need to add additional tokens to the dictionary, this is likely the right
        place to do it.
        """
        d = self.dictionary_class()(self.opt)
        d.add_extra_special_tokens(self._get_special_tokens())

        return d

    def _resize_token_embeddings(self):
        """
        Must define this for your agent

        Must make a call to resize the token embeddings and load the model state dict.
        """
        raise RuntimeError('Must define this funciton for your specific agent.')

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        Override from Torch Agent to resize the token embeddings.s
        """
        try:
            self.model.load_state_dict(state_dict)
            return False
        except RuntimeError as msg:
            msg_ = str(msg)
            if 'size mismatch' in msg_ and 'embedding' in msg_:
                self._resize_token_embeddings(state_dict, msg_)
                return True  # resized
            else:
                raise (msg)

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Override: do not load optimizer state if resized.
        """
        if hasattr(self, 'resized') and self.resized:
            optim_states = None
            logging.warn('Not loading optimizer due to resize in token embeddings')

        super().init_optim(params, optim_states, saved_optim_type)

    def load(self, path: str) -> Dict[str, Any]:
        """
        Return opt and model states.

        Override this method to catch a resize
        """
        import parlai.utils.pickle

        states = torch.load(
            path, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
        )
        self.resized = False
        if 'model' in states:
            self.resized = self.load_state_dict(states['model'])
        if 'optimizer' in states and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(states['optimizer'])

        return states

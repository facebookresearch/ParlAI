# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .transformer import TransformerRankerAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
import torch


class BiencoderAgent(TransformerRankerAgent):
    """
    Equivalent of bert_ranker/biencoder but does not rely on an external
    library (hugging face).
    """

    def vectorize(self, *args, **kwargs):
        """ Add the start and end token to the text.
        """
        kwargs['add_start'] = True
        kwargs['add_end'] = True
        obs = TorchRankerAgent.vectorize(self, *args, **kwargs)
        return obs

    def _set_text_vec(self, *args, **kwargs):
        """ Add the start and end token to the text.
        """
        obs = super()._set_text_vec(*args, **kwargs)
        if 'text_vec' in obs and 'added_start_end_tokens' not in obs:
            obs.force_set(
                'text_vec', self._add_start_end_tokens(obs['text_vec'], True, True)
            )
            obs['added_start_end_tokens'] = True
        return obs

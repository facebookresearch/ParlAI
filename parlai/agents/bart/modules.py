#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART Module.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Any, Dict, Union, List, Optional

from parlai.agents.transformer.modules import TransformerGeneratorModel


class BartModel(TransformerGeneratorModel):
    """
    BART Model.
    """

    def output(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute output logits.

        Override standard TGM output to _not_ prevent generation of BOS.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        return output

    def decode_forced(
        self, encoder_states: Tuple[Any, ...], ys: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat(
            [
                torch.LongTensor([self.END_IDX]).detach().expand(bsz, 1).to(inputs),
                inputs,
            ],
            1,
        )
        latent, _ = self.decoder(inputs, encoder_states)
        logits = self.output(latent)
        _, preds = logits.max(dim=2)
        return logits, preds

    def reorder_decoder_incremental_state(
        self,
        incremental_state: Dict[str, Any],
        inds: Union[List[int], torch.LongTensor],
    ) -> Optional[Dict[str, Any]]:
        """
        Incremental state is too hard with all the docs and what not.

        We leave as a future exercise.
        """
        return None
#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART Module.
"""
import torch
import torch.nn.functional as F

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

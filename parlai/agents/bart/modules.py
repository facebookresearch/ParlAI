#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART Module.
"""
from parlai.agents.transformer.modules import TransformerGeneratorModel
import torch.nn.functional as F


class BartModel(TransformerGeneratorModel):
    """
    BART Model.
    """

    def output(self, tensor):
        """
        Compute output logits.
        """
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        return output

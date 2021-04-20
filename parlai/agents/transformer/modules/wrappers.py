#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from parlai.agents.transformer.modules import TransformerEncoder


class TransformerResponseWrapper(nn.Module):
    """
    Wrap transformer response.

    Pushes input through transformer and MLP.
    """

    def __init__(self, transformer: TransformerEncoder, hdim: int):
        super(TransformerResponseWrapper, self).__init__()
        dim = transformer.out_dim
        self.transformer = transformer
        self.mlp = nn.Sequential(
            nn.Linear(dim, hdim),
            nn.ReLU(),  # TODO: should this also be gelu?
            nn.Linear(hdim, dim),
        )

    def forward(self, *args) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.mlp(self.transformer(*args))


class TransformerLinearWrapper(nn.Module):
    """
    Wrap a transformer in a linear layer.
    """

    def __init__(self, transformer: TransformerEncoder, output_dim: int):
        super().__init__()
        self.transformer = transformer
        input_dim = transformer.out_dim
        self.additional_linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, *args) -> torch.Tensor:
        """
        Forward pass.

        Apply transformer, then additional linear layer.
        """
        context_h = self.transformer(*args)
        return self.additional_linear_layer(context_h)

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Feedforward neural network, as used in transformer implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerFFN(nn.Module):
    """
    Implements the FFN part of the transformer.
    """

    def __init__(
        self,
        dim: int,
        dim_hidden: int,
        relu_dropout: float = 0,
        activation: str = 'relu',
    ):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        if activation == 'relu':
            self.nonlinear = F.relu
        elif activation == 'gelu':
            self.nonlinear = F.gelu
        else:
            raise ValueError(
                "Don't know how to handle --activation {}".format(activation)
            )
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x

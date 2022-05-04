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

from parlai.core.opt import Opt
from parlai.core.params import default


class TransformerFFN(nn.Module):
    """
    Implements the FFN part of the transformer.
    """

    def __init__(
        self,
        opt: Opt,
        dim: int = None,
        dim_hidden: int = None,
        relu_dropout: float = 0,
        activation: str = 'relu',
        **kwargs,
    ):
        super(TransformerFFN, self).__init__(**kwargs)

        dim = default(dim, opt['embedding_size'])
        dim_hidden = default(dim_hidden, opt['ffn_size'])

        self.opt = opt
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.nonlinear(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x

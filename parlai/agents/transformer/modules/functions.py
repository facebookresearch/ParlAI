#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Pure functions used in transformer implementation.
"""

import numpy as np
import torch
import torch.nn as nn

from parlai.core.opt import Opt


def create_position_codes(n_pos, dim, out):
    """
    Create positional codes and store them in ``out``.
    """
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
            for pos in range(n_pos)
        ]
    )

    out.detach_()
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)


def create_embeddings(dictionary, embedding_size, padding_idx):
    """
    Create and initialize word embeddings.
    """
    e = nn.Embedding(len(dictionary), embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size**-0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


def get_n_positions_from_options(opt: Opt):
    """
    Determine n_positions from options dict.
    """
    if opt.get('n_positions'):
        # if the number of positions is explicitly provided, use that
        n_positions = opt['n_positions']
    else:
        # else, use the worst case from truncate
        n_positions = max(
            opt.get('truncate') or 0,
            opt.get('text_truncate') or 0,
            opt.get('label_truncate') or 0,
        )
        if n_positions == 0:
            # default to 1024
            n_positions = 1024
    if n_positions < 0:
        raise ValueError('n_positions must be positive')
    return n_positions

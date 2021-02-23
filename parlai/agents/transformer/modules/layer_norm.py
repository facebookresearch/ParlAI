#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Layer norm, as used by transformer.
"""

import torch
from typing import List, Union

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as _LayerNorm

    APEX_LAYER_NORM = True
except ImportError:
    from torch.nn import _LayerNorm

    APEX_LAYER_NORM = False

LAYER_NORM_EPS = 1e-5  # Epsilon for layer norm.


def LayerNorm(normalized_shape: Union[int, List[int], torch.Size]) -> _LayerNorm:
    return _LayerNorm(normalized_shape=normalized_shape, eps=LAYER_NORM_EPS)


def normalize(tensor, norm_layer):
    """
    Broadcast layer norm.
    """
    is_cpu = tensor.device == 'cpu' or tensor.device.type == 'cpu'
    if APEX_LAYER_NORM and not is_cpu:
        # fused_layer_norm has a bug around multi-device networks.
        # https://github.com/NVIDIA/apex/issues/770
        # https://github.com/NVIDIA/apex/issues/371
        with torch.cuda.device(tensor.device):
            return norm_layer(tensor)
    else:
        return norm_layer(tensor)

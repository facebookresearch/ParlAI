#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Definitions of general ParlAI types.
"""
from typing import TypeVar


class ParlAITypes(object):
    """
    Typing definitions of general ParlAI structures.

    Attributes:
        TShared: type of the "shared" structure in ParlAI
    """

    TShared = TypeVar('TShared', bound=dict)

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum


def wrap_content_begin_token(content_type: str):
    """
    Generates the token that signals the beggining of the span for a particular feature.
    """
    return f'__{content_type}__'


def wrap_content_end_token(content_type: str):
    """
    Generates the token that signals the end of the span for a particular feature.
    """
    return f'__end-{content_type}__'


class GraphMutations(Enum):
    NO_MUT = 0
    DEL = 1
    ADD = 2


EMPTY_GRAPH_TOKEN = '__empty-graph__'

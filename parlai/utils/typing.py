#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Definitions of general ParlAI types.
"""
from typing import TypeVar, Dict, Any


class _Shared(Dict[str, Any]):
    """
    ParlAI ``shared`` Structure.

    The `shared` dict that is used to instantiate shared agents in ParlAI,
    e.g. when using batching, distributed training, etc.

    Type is ``TShared``.
    """


TShared = TypeVar('TShared', bound=_Shared)

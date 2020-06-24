#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities related to handling data.
"""


def is_training(datatype: str) -> bool:
    """
    Return whether we should randomize given datatype.

    :param datatype:
        parlai datatype

    :return is_training:
        given datatype, return whether we're in a random sampling state.
    """
    assert datatype is not None, 'datatype must not be none'
    return 'train' in datatype and 'evalmode' not in datatype
